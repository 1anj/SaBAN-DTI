import json
import sys
import os
import re
import torch
import logging
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import selfies
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

sys.path.append("../")

LOGGER = logging.getLogger(__name__)

class MERGEDDatasetProcessor:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.name = "MERGED"
        self.seed = getattr(args, 'seed', 42)
        self.token_cache = getattr(args, 'token_cache', 'dataset/processed_token')
        self.neg_sample_ratio = getattr(args, 'neg_sample_ratio', 1)
        self.exclusion_file = getattr(args, 'exclusion_file', None)
        self.train_only_excluded = getattr(args, 'train_only_excluded', False)
        
        self.protein_filter_file = getattr(args, 'protein_filter_file', None)
        self.protein_filter_type = getattr(args, 'protein_filter_type', None)
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.9)
        
        self.csv_file = os.path.join(self.data_path, "MERGED_dataset.csv")
        self.config_file = os.path.join(self.token_cache, "MERGED", "dataset_config.json")
        
        self.merged_dir = os.path.join(self.data_path, 'huge_data')
        
        print("Loading ID mappings...")
        self.id_to_sequence = np.load(os.path.join(self.merged_dir, 'id_to_sequence.npy'), allow_pickle=True).item()
        self.id_to_saprot_sequence = np.load(os.path.join(self.merged_dir, 'id_to_saprot_sequence.npy'), allow_pickle=True).item()
        self.id_to_smiles = np.load(os.path.join(self.merged_dir, 'id_to_smiles.npy'), allow_pickle=True).item()
        
        self.exclusion = set()
        if self.exclusion_file:
            exclusion_path = os.path.join(self.merged_dir, self.exclusion_file) if not os.path.isabs(self.exclusion_file) else self.exclusion_file
            if os.path.exists(exclusion_path):
                with open(exclusion_path, 'r') as f:
                    for line in f:
                        self.exclusion.add(line.strip())
                print(f"Loaded {len(self.exclusion)} exclusion IDs")
        
        self.target_proteins = None
        if self.protein_filter_file and self.protein_filter_type:
            self.target_proteins = self._load_target_proteins()
            if self.target_proteins:
                print(f"Loaded {len(self.target_proteins)} target proteins from {self.protein_filter_type} dataset")
        
        id_list = []
        
        if self.train_only_excluded:
            print(f"TRAIN_ONLY_EXCLUDED MODE: Training ONLY on {len(self.exclusion)} excluded proteins")
        elif self.exclusion:
            print(f"EXCLUSION MODE: Excluding {len(self.exclusion)} proteins from training")
        
        for k in list(self.id_to_sequence.keys()):
            if self.train_only_excluded:
                if k not in self.exclusion:
                    continue
            else:
                if k in self.exclusion:
                    continue
            
            sequence = self.id_to_sequence[k]
            if any(char.isdigit() for char in sequence):
                continue
            
            if self.target_proteins:
                if not self._filter_by_similarity(k, sequence):
                    continue
            
            id_list.append(k)
        
        self.valid_target_ids = set(id_list)
        
        total_proteins = len(self.id_to_sequence)
        if self.target_proteins:
            print(f"Applied k-mer Jaccard similarity filtering with threshold > {self.similarity_threshold}")
        if self.train_only_excluded:
            print(f"Filtered to {len(self.valid_target_ids)} valid target IDs from {len(self.exclusion)} excluded proteins")
        else:
            print(f"Valid target IDs: {len(self.valid_target_ids)} (from {total_proteins} total, excluded {len(self.exclusion)})")
        
        self.id_to_smiles = {str(k): v for k, v in self.id_to_smiles.items()}
        
        self._load_splits()
        
        self.prot_col = 'Seq'
        self.drug_col = 'selfies'
        self.has_raw_protein = True
        self.has_raw_smiles = True
        
        print(f"Loaded MERGED dataset:")
        print(f"  Train: {len(self.train_dataset_df)} samples")
        print(f"  Valid: {len(self.val_dataset_df)} samples")
        print(f"  Test: {len(self.test_dataset_df)} samples")
        
    def _process_batch(self, batch_data, valid_target_ids, id_to_sequence, id_to_saprot_sequence, id_to_smiles):
        results = []
        for idx, row in batch_data:
            ligand_id = str(row['ligand'])
            protein_id = str(row['aa_seq'])
            
            if protein_id not in valid_target_ids:
                continue
            
            protein_seq = id_to_sequence.get(protein_id, '')
            saprot_seq = id_to_saprot_sequence.get(protein_id, '')
            
            if not saprot_seq:
                saprot_seq = protein_seq
            if not saprot_seq:
                continue
            
            smiles = id_to_smiles.get(ligand_id, '')
            if not smiles:
                continue
            
            try:
                selfies_str = selfies.encoder(smiles)
                if not selfies_str:
                    continue
            except:
                continue
            
            results.append((idx, protein_seq, saprot_seq, smiles, selfies_str))
        return results
    
    def _load_splits(self):
        splits = ['train', 'val', 'test']
        
        n_workers = min(cpu_count() - 1, 8)
        
        for split in splits:
            pos_file = os.path.join(self.merged_dir, f'merged_pos_uniq_{split}_rand.tsv')
            neg_file = os.path.join(self.merged_dir, f'merged_neg_uniq_{split}_rand.tsv')
            
            print(f"Loading {split} split...")
            pos_df = pd.read_csv(pos_file, sep='\t')
            pos_df['label'] = 1
            
            neg_df = pd.read_csv(neg_file, sep='\t')
            neg_df['label'] = 0
            
            print(f"  Loaded {len(pos_df)} positive and {len(neg_df)} negative samples")
            
            if self.train_only_excluded or self.exclusion:
                pos_before = len(pos_df)
                neg_before = len(neg_df)
                
                pos_df['aa_seq'] = pos_df['aa_seq'].astype(str)
                neg_df['aa_seq'] = neg_df['aa_seq'].astype(str)
                
                pos_df = pos_df[pos_df['aa_seq'].isin(self.valid_target_ids)]
                neg_df = neg_df[neg_df['aa_seq'].isin(self.valid_target_ids)]
                
                print(f"  After filtering: {len(pos_df)} positive (from {pos_before}) and {len(neg_df)} negative (from {neg_before}) samples")
                
                if len(pos_df) == 0 and len(neg_df) == 0:
                    print(f"  WARNING: No samples left after filtering for {split} split!")
                    df = pd.DataFrame(columns=['ligand', 'aa_seq', 'label', 'Protein', 'Seq', 'SMILES', 'selfies'])
                    if split == 'val':
                        setattr(self, 'val_dataset_df', df)
                        setattr(self, 'valid_dataset_df', df)
                    else:
                        setattr(self, f'{split}_dataset_df', df)
                    
                    loader_name = 'val_data_loader' if split == 'val' else f'{split}_data_loader'
                    setattr(self, loader_name, DTIDataset(([], [], [], [], [])))
                    continue
            
            if self.neg_sample_ratio == -1:
                print(f"  Using all {len(neg_df)} negative samples (actual ratio: {len(neg_df)/len(pos_df):.2f}:1)")
            elif self.neg_sample_ratio > 0 and self.neg_sample_ratio < len(neg_df) / len(pos_df):
                n_neg_samples = min(int(len(pos_df) * self.neg_sample_ratio), len(neg_df))
                neg_df = neg_df.sample(n=n_neg_samples, random_state=self.seed)
                print(f"  Sampled {n_neg_samples} negative samples to match ratio {self.neg_sample_ratio}:1")
            else:
                print(f"  Using all {len(neg_df)} negative samples (requested ratio {self.neg_sample_ratio}:1 exceeds available)")
            
            df = pd.concat([pos_df, neg_df], ignore_index=True)
            print(f"  Combined dataset: {len(df)} total samples ({len(pos_df)} pos + {len(neg_df)} neg)")
            
            chunk_size = max(100, len(df) // (n_workers * 4))
            chunks = []
            for i in range(0, len(df), chunk_size):
                chunk = [(idx, row) for idx, row in df.iloc[i:i+chunk_size].iterrows()]
                chunks.append(chunk)
            
            print(f"  Processing with {n_workers} workers in {len(chunks)} chunks...")
            
            process_func = partial(self._process_batch,
                                  valid_target_ids=self.valid_target_ids,
                                  id_to_sequence=self.id_to_sequence,
                                  id_to_saprot_sequence=self.id_to_saprot_sequence,
                                  id_to_smiles=self.id_to_smiles)
            
            with Pool(n_workers) as pool:
                chunk_results = pool.map(process_func, chunks)
            
            sequences = []
            saprot_sequences = []
            smiles_list = []
            selfies_list = []
            valid_rows = []
            
            for chunk_result in chunk_results:
                for idx, protein_seq, saprot_seq, smiles, selfies_str in chunk_result:
                    valid_rows.append(idx)
                    sequences.append(protein_seq)
                    saprot_sequences.append(saprot_seq)
                    smiles_list.append(smiles)
                    selfies_list.append(selfies_str)
            
            if self.train_only_excluded:
                print(f"  Processed {len(valid_rows)} valid samples (filtered to ONLY excluded proteins)")
                print(f"    Note: Only proteins from exclusion list that appear in {split} split are included")
            elif self.exclusion:
                print(f"  Processed {len(valid_rows)} valid samples (excluding {len(self.exclusion)} proteins)")
            else:
                print(f"  Processed {len(valid_rows)} valid samples")
            
            df = df.loc[valid_rows].reset_index(drop=True)
            
            df['Protein'] = sequences
            df['Seq'] = saprot_sequences
            df['SMILES'] = smiles_list
            df['selfies'] = selfies_list
            
            if self.train_only_excluded or self.exclusion:
                unique_proteins = df['aa_seq'].nunique() if 'aa_seq' in df.columns else len(set(sequences))
                unique_drugs = df['ligand'].nunique() if 'ligand' in df.columns else len(set(smiles_list))
                print(f"  Dataset contains {unique_proteins} unique proteins and {unique_drugs} unique drugs")
            
            if split == 'val':
                setattr(self, 'val_dataset_df', df)
                setattr(self, 'valid_dataset_df', df)
            else:
                setattr(self, f'{split}_dataset_df', df)
            
            prot_seqs = df['Seq'].values
            drug_seqs = df['selfies'].values
            raw_prot = df['Protein'].values
            raw_smiles = df['SMILES'].values
            
            loader_name = 'val_data_loader' if split == 'val' else f'{split}_data_loader'
            setattr(self, loader_name, DTIDataset((
                prot_seqs, drug_seqs, df["label"].values, raw_prot, raw_smiles
            )))
    
    def set_fold(self, fold_idx):
        pass
    
    def get_num_folds(self):
        return 1
    
    def get_train_examples(self):
        return self.train_data_loader
    
    def get_val_examples(self):
        return self.val_data_loader
    
    def get_test_examples(self):
        return self.test_data_loader
    
    def get_fold_info(self):
        return [{
            "fold": 0,
            "train_size": len(self.train_dataset_df),
            "valid_size": len(self.val_dataset_df),
            "test_size": len(self.test_dataset_df),
            "features_exist": False,
            "features_saved_in_config": False
        }]
    
    def _load_target_proteins(self):
        import json
        
        target_proteins = {}
        
        if self.protein_filter_type == "DUDE":
            if os.path.exists(self.protein_filter_file):
                with open(self.protein_filter_file, 'r') as f:
                    current_name = None
                    current_seq = []
                    for line in f:
                        if line.startswith('>'):
                            if current_name:
                                target_proteins[current_name] = ''.join(current_seq)
                            current_name = line.strip().split()[0][1:]
                            current_seq = []
                        else:
                            current_seq.append(line.strip())
                    if current_name:
                        target_proteins[current_name] = ''.join(current_seq)
                print(f"Loaded {len(target_proteins)} DUDE proteins")
                
        elif self.protein_filter_type == "PCBA":
            if os.path.exists(self.protein_filter_file):
                with open(self.protein_filter_file, 'r') as f:
                    pcba_data = json.load(f)
                    for protein_name, sequences in pcba_data.items():
                        if sequences and len(sequences) > 0:
                            target_proteins[protein_name] = sequences[0]
                print(f"Loaded {len(target_proteins)} PCBA proteins")
        
        return target_proteins
    
    def _calculate_kmer_similarity(self, seq1, seq2, k=3):
        if not seq1 or not seq2:
            return 0.0
        
        def get_kmers(sequence, k):
            kmers = set()
            if len(sequence) >= k:
                for i in range(len(sequence) - k + 1):
                    kmers.add(sequence[i:i+k])
            return kmers
        
        kmers1 = get_kmers(seq1, k)
        kmers2 = get_kmers(seq2, k)
        
        if not kmers1 and not kmers2:
            return 1.0 if seq1 == seq2 else 0.0
        if not kmers1 or not kmers2:
            return 0.0
        
        intersection = kmers1.intersection(kmers2)
        union = kmers1.union(kmers2)
        
        if len(union) == 0:
            return 0.0
        
        jaccard_index = len(intersection) / len(union)
        
        return jaccard_index
    
    def _filter_by_similarity(self, protein_id, protein_seq):
        if not self.target_proteins:
            return True
        
        for target_name, target_seq in self.target_proteins.items():
            similarity = self._calculate_kmer_similarity(protein_seq, target_seq)
            if similarity > self.similarity_threshold:
                return True
        
        return False
    
    def check_processed_features_exist(self, fold_idx):
        dataset_name = "MERGED"
        fold_path = os.path.join(self.token_cache, dataset_name, f"fold_1")
        
        if not os.path.exists(fold_path):
            return False
        
        try:
            files = os.listdir(fold_path)
            return all(
                any(f.startswith(f"{dataset_name}_{split}_batch") for f in files)
                for split in ['train', 'valid', 'test']
            )
        except OSError:
            return False
    
    def update_feature_status(self, fold_idx, saved=True):
        print(f"Feature status for MERGED dataset: {'saved' if saved else 'not saved'}")
    
    def update_epoch_data(self, split='train'):
        if split != 'train':
            return
        
        pos_file = os.path.join(self.merged_dir, f'merged_pos_uniq_train_rand.tsv')
        neg_file = os.path.join(self.merged_dir, f'merged_neg_uniq_train_rand.tsv')
        
        pos_df = pd.read_csv(pos_file, sep='\t')
        pos_df['label'] = 1
        
        neg_df = pd.read_csv(neg_file, sep='\t')
        neg_df['label'] = 0
        
        if self.neg_sample_ratio < len(neg_df) / len(pos_df):
            n_neg_samples = min(int(len(pos_df) * self.neg_sample_ratio), len(neg_df))
            import random
            neg_df = neg_df.sample(n=n_neg_samples, random_state=random.randint(0, 10000))
        
        print(f"Resampled training data with {len(pos_df)} positive and {len(neg_df)} negative samples")

class DatasetProcessor:
    def __init__(self, args):
        self.args = args
        self.csv_file = args.data_path
        self.name = os.path.basename(args.data_path).replace('.csv', '')
        self.k_folds = getattr(args, 'k_folds', 5)
        self.seed = getattr(args, 'seed', 42)
        self.stratified = getattr(args, 'stratified', True)
        self.token_cache = getattr(args, 'token_cache', 'dataset/processed_token')
            
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
            
        self.full_dataset_df = pd.read_csv(self.csv_file)
        print(f"Loaded dataset with {len(self.full_dataset_df)} samples")
        
        initial_len = len(self.full_dataset_df)
        
        has_structure_aware = 'Seq' in self.full_dataset_df.columns and 'selfies' in self.full_dataset_df.columns
        has_standard = 'Protein' in self.full_dataset_df.columns and 'SMILES' in self.full_dataset_df.columns
        
        if has_structure_aware:
            self.prot_col = 'Seq'
            self.drug_col = 'selfies'
            print("Using structure-aware sequences (Seq) and SELFIES")
        elif has_standard:
            self.prot_col = 'Protein'
            self.drug_col = 'SMILES'
            print("Using standard protein sequences and SMILES")
        else:
            raise ValueError(f"Missing required columns. Expected either ('Seq', 'selfies') or ('Protein', 'SMILES'). Available: {list(self.full_dataset_df.columns)}")
        
        self.has_raw_protein = 'Protein' in self.full_dataset_df.columns
        self.has_raw_smiles = 'SMILES' in self.full_dataset_df.columns
        
        self.full_dataset_df = self.full_dataset_df.dropna(subset=[self.prot_col, self.drug_col, 'label'])
        self.full_dataset_df = self.full_dataset_df[
            (self.full_dataset_df[self.prot_col].str.strip() != '') & 
            (self.full_dataset_df[self.drug_col].str.strip() != '') & 
            (self.full_dataset_df['label'].astype(str).str.strip() != '')
        ].reset_index(drop=True)
        
        if len(self.full_dataset_df) < initial_len:
            print(f"Cleaned dataset: {initial_len} → {len(self.full_dataset_df)} samples")
        
        self.config_file = self._get_config_path()
        
        if os.path.exists(self.config_file):
            self._load_fold_config()
        else:
            self._create_and_save_fold_config()
        
        self.current_fold = 0
        
    def _get_dataset_name(self):
        return os.path.basename(self.csv_file).replace('.csv', '')
    
    def _get_config_path(self):
        dataset_name = self._get_dataset_name()
        config_dir = os.path.join(self.token_cache, dataset_name)
        config_type = 'stratified' if self.stratified else 'regular'
        return os.path.join(config_dir, f"kfold_config_k{self.k_folds}_seed{self.seed}_{config_type}.json")
    
    def _compute_dataset_hash(self):
        hash_content = f"{len(self.full_dataset_df)}{self.full_dataset_df[self.prot_col].str.cat()}{self.full_dataset_df[self.drug_col].str.cat()}{self.full_dataset_df['label'].sum()}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def _create_and_save_fold_config(self):
        print(f"Creating k-fold configuration...")
        self.fold_indices = self._create_kfold_splits()
        
        config = {
            "metadata": {
                "dataset_path": self.csv_file,
                "total_samples": len(self.full_dataset_df),
                "k_folds": self.k_folds,
                "seed": self.seed,
                "stratified": self.stratified,
                "dataset_hash": self._compute_dataset_hash(),
                "created_at": datetime.now().isoformat(),
                "split_ratio": "7:1:2"
            },
            "folds": {},
            "features_saved": {}
        }
        
        for i, fold in enumerate(self.fold_indices):
            fold_key = f"fold_{i}"
            config["folds"][fold_key] = {
                "train": fold["train"].tolist(),
                "valid": fold["valid"].tolist(),
                "test": fold["test"].tolist(),
                "train_size": len(fold["train"]),
                "valid_size": len(fold["valid"]),
                "test_size": len(fold["test"])
            }
            config["features_saved"][fold_key] = False
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved {self.k_folds}-fold config: {len(self.full_dataset_df)} samples")
    
    def update_feature_status(self, fold_idx, saved=True):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            if "features_saved" not in config:
                config["features_saved"] = {}
            
            config["features_saved"][f"fold_{fold_idx}"] = saved
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Updated feature status for fold {fold_idx + 1}: {'saved' if saved else 'not saved'}")
    
    def _load_fold_config(self):
        print(f"Loading k-fold configuration from {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        current_hash = self._compute_dataset_hash()
        saved_hash = config["metadata"]["dataset_hash"]
        
        if current_hash != saved_hash:
            print(f"WARNING: Dataset has changed since configuration was created!")
            print(f"Current samples: {len(self.full_dataset_df)}, Saved samples: {config['metadata']['total_samples']}")
            print(f"Regenerating fold configuration...")
            self._create_and_save_fold_config()
            return
        
        if (config["metadata"]["k_folds"] != self.k_folds or 
            config["metadata"]["seed"] != self.seed or 
            config["metadata"]["stratified"] != self.stratified):
            print(f"WARNING: Parameters have changed since configuration was created!")
            print(f"Regenerating fold configuration with new parameters...")
            self._create_and_save_fold_config()
            return
        
        self.fold_indices = []
        for i in range(config["metadata"]["k_folds"]):
            fold_key = f"fold_{i}"
            self.fold_indices.append({
                "train": np.array(config["folds"][fold_key]["train"]),
                "valid": np.array(config["folds"][fold_key]["valid"]),
                "test": np.array(config["folds"][fold_key]["test"])
            })
        
        print(f"Successfully loaded k-fold configuration:")
        print(f"  - Created at: {config['metadata']['created_at']}")
        print(f"  - Total samples: {config['metadata']['total_samples']}")
        print(f"  - K-folds: {config['metadata']['k_folds']}")
        print(f"  - Stratified: {config['metadata']['stratified']}")
        print(f"  - Split ratio: {config['metadata']['split_ratio']}")
    
    def _create_kfold_splits(self):
        if self.stratified:
            kf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            labels = self.full_dataset_df['label'].values
        else:
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            labels = None
            
        fold_indices = []
        for train_idx, test_idx in kf.split(self.full_dataset_df, labels):
            if self.stratified:
                train_labels = self.full_dataset_df.iloc[train_idx]['label'].values
                train_train_idx, val_idx = train_test_split(
                    train_idx, 
                    test_size=0.125,
                    stratify=train_labels,
                    random_state=self.seed
                )
            else:
                train_train_idx, val_idx = train_test_split(
                    train_idx, 
                    test_size=0.125,
                    random_state=self.seed
                )
                
            fold_indices.append({
                'train': train_train_idx,
                'valid': val_idx,
                'test': test_idx
            })
            
        return fold_indices
    
    def set_fold(self, fold_idx):
        if fold_idx >= self.k_folds:
            raise ValueError(f"Fold {fold_idx} exceeds {self.k_folds} folds")
        
        self.current_fold = fold_idx
        indices = self.fold_indices[fold_idx]
        
        for name, idx_key in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
            df = self.full_dataset_df.iloc[indices[idx_key]].reset_index(drop=True)
            setattr(self, f"{name}_dataset_df", df)
            
            prot_seqs = df[self.prot_col].values
            drug_seqs = df[self.drug_col].values
            
            if self.has_raw_protein:
                raw_prot = df['Protein'].values
            else:
                raw_prot = df[self.prot_col].values
                
            if self.has_raw_smiles:
                raw_smiles = df['SMILES'].values
            else:
                raw_smiles = []
                for drug in drug_seqs:
                    try:
                        if drug.startswith('[') and drug.endswith(']'):
                            smiles = selfies.decoder(drug)
                            raw_smiles.append(smiles)
                        else:
                            raw_smiles.append(drug)
                    except:
                        raw_smiles.append(drug)
                raw_smiles = np.array(raw_smiles)
            
            setattr(self, f"{name}_data_loader", DTIDataset((
                prot_seqs, drug_seqs, df["label"].values, raw_prot, raw_smiles
            )))
        
    
    def get_num_folds(self):
        return self.k_folds
    
    def check_processed_features_exist(self, fold_idx):
        dataset_name = self._get_dataset_name()
        fold_path = os.path.join(self.token_cache, dataset_name, f"fold_{fold_idx + 1}")
        
        if not os.path.exists(fold_path):
            return False
        
        try:
            files = os.listdir(fold_path)
            return all(
                any(f.startswith(f"{dataset_name}_{split}_batch") for f in files)
                for split in ['train', 'valid', 'test']
            )
        except OSError:
            return False
    
    def get_fold_info(self):
        features_saved = {}
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                features_saved = json.load(f).get("features_saved", {})
            
        return [{
            "fold": i,
            "train_size": len(self.fold_indices[i]["train"]),
            "valid_size": len(self.fold_indices[i]["valid"]),
            "test_size": len(self.fold_indices[i]["test"]),
            "features_exist": self.check_processed_features_exist(i),
            "features_saved_in_config": features_saved.get(f"fold_{i}", False)
        } for i in range(self.k_folds)]
    
    def get_train_examples(self):
        return self.train_data_loader
    
    def get_val_examples(self):
        return self.val_data_loader
    
    def get_test_examples(self):
        return self.test_data_loader
    

class DTIDataset(Dataset):
    def __init__(self, data_examples):
        if len(data_examples) == 5:
            self.prot_seqs, self.drug_seqs, self.scores, self.raw_prot, self.raw_smiles = data_examples
            self.has_raw = True
        else:
            self.prot_seqs, self.drug_seqs, self.scores = data_examples
            self.raw_prot = None
            self.raw_smiles = None
            self.has_raw = False

    def __getitem__(self, idx):
        return (self.prot_seqs[idx],
                self.drug_seqs[idx],
                self.scores[idx],
                self.raw_prot[idx] if self.raw_prot is not None else self.prot_seqs[idx],
                self.raw_smiles[idx] if self.raw_smiles is not None else self.drug_seqs[idx])

    def __len__(self):
        return len(self.prot_seqs)

class BatchFileDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx], map_location='cpu')
        prot_seqs = data.get('prot_seqs', None)
        drug_smiles = data.get('drug_smiles', None)
        
        if prot_seqs is not None and drug_smiles is not None:
            return (data['prot'], data['drug'], data['prot_mask'], 
                   data['drug_mask'], data['y'], prot_seqs, drug_smiles)
        else:
            return (data['prot'], data['drug'], data['prot_mask'], 
                   data['drug_mask'], data['y'])
    
    def cleanup(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class DrugTokenizer:
    def __init__(self, vocab_path="data/Tokenizer/vocab.json", special_tokens_path="data/Tokenizer/special_tokens_map.json"):
        self.vocab, self.special_tokens = self.load_vocab_and_special_tokens(vocab_path, special_tokens_path)
        self.cls_token_id = self.vocab[self.special_tokens['cls_token']]
        self.sep_token_id = self.vocab[self.special_tokens['sep_token']]
        self.unk_token_id = self.vocab[self.special_tokens['unk_token']]
        self.pad_token_id = self.vocab[self.special_tokens['pad_token']]
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def load_vocab_and_special_tokens(self, vocab_path, special_tokens_path):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
        with open(special_tokens_path, 'r', encoding='utf-8') as special_tokens_file:
            special_tokens_raw = json.load(special_tokens_file)

        special_tokens = {key: value['content'] for key, value in special_tokens_raw.items()}
        return vocab, special_tokens

    def encode(self, sequence):
        tokens = re.findall(r'\[([^\[\]]+)\]', sequence)
        input_ids = [self.cls_token_id] + [self.vocab.get(token, self.unk_token_id) for token in tokens] + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def batch_encode_plus(self, sequences, max_length, **_):
        input_ids_list = []
        attention_mask_list = []

        for sequence in sequences:
            encoded = self.encode(sequence)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            elif len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                input_ids = input_ids + [self.vocab[self.special_tokens['pad_token']]] * pad_length
                attention_mask = attention_mask + [0] * pad_length

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        return {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
        }

    def decode(self, input_ids, skip_special_tokens=False):
        tokens = []
        for id in input_ids:
            if skip_special_tokens and id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                continue
            tokens.append(self.id_to_token.get(id, self.special_tokens['unk_token']))
        sequence = ''.join([f'[{token}]' for token in tokens])
        return sequence

def load_dataset(args):
    data_path = args.data_path
    
    if 'MERGED' in data_path or os.path.exists(os.path.join(data_path, 'huge_data')):
        print("Detected MERGED dataset format")
        return MERGEDDatasetProcessor(args)
    else:
        return DatasetProcessor(args)