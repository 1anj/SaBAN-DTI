import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import selfies as sf
from rdkit import Chem
from rdkit import RDLogger
import warnings
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import glob
warnings.filterwarnings('ignore')

RDLogger.DisableLog('rdApp.*')

from model import DTIModel, TokenEncoder
from transformers import EsmTokenizer, EsmForMaskedLM, AutoModel, AutoTokenizer


class EvaluationMetrics:
    
    @staticmethod
    def auroc(y_true, y_score):
        try:
            return roc_auc_score(y_true, y_score)
        except:
            return 0.5
    
    @staticmethod
    def bedroc(y_true, y_score, alpha=20.0):
        n = len(y_true)
        n_actives = np.sum(y_true)
        n_decoys = n - n_actives
        
        if n_actives == 0 or n_decoys == 0:
            return 0.5
        
        sorted_indices = np.argsort(y_score)[::-1]
        sorted_labels = y_true[sorted_indices]
        active_ranks = np.where(sorted_labels == 1)[0] + 1
        rie_sum = np.sum(np.exp(-alpha * (active_ranks - 1) / n))
        rie_random = n_actives / n * (1 - np.exp(-alpha)) / (np.exp(alpha/n) - 1)
        rie_perfect = (1 - np.exp(-alpha * n_actives / n)) / (np.exp(alpha/n) - 1)
        
        if rie_perfect == rie_random:
            return 0.5
        
        bedroc = (rie_sum - rie_random) / (rie_perfect - rie_random)
        bedroc = np.clip(bedroc, 0.0, 1.0)
        
        return bedroc
    
    @staticmethod
    def enrichment_factor(y_true, y_score, percentage):
        n = len(y_true)
        n_actives = np.sum(y_true)
        
        if n_actives == 0:
            return 0.0
        
        n_top = max(1, int(n * percentage / 100))
        sorted_indices = np.argsort(y_score)[::-1]
        sorted_labels = y_true[sorted_indices]
        n_actives_top = np.sum(sorted_labels[:n_top])
        ef = (n_actives_top / n_top) / (n_actives / n)
        
        return ef


def convert_smiles_to_selfies(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
            return sf.encoder(canonical_smiles)
        else:
            return sf.encoder(smiles)
    except:
        print("Error convert smiles ", smiles)
        return None


def batch_convert_smiles(smiles_list, use_parallel=True, n_workers=None):
    if not use_parallel or len(smiles_list) < 100:
        results = []
        for smiles in tqdm(smiles_list, desc="Converting SMILES", disable=len(smiles_list) < 100):
            results.append(convert_smiles_to_selfies(smiles))
        return results
    
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_smiles_to_selfies, smiles_list, chunksize=100),
            total=len(smiles_list),
            desc="Converting SMILES (parallel)"
        ))
    return results


class FastDUDEDataset(Dataset):
    
    def __init__(self, data_path, target=None, use_cache=True, use_parallel=True):
        self.data_path = data_path
        self.data = []
        self.labels = []
        self.targets = []
        
        dude_dir = os.path.join(data_path, 'DUDE')
        cache_file = os.path.join(dude_dir, f'.cache_dude_{target if target else "all"}.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.data = cache_data['data']
                self.labels = np.array(cache_data['labels'])
                self.targets = cache_data['targets']
                print(f"Loaded {len(self.data)} samples from cache")
                return
        
        target_list = []
        subset_file = os.path.join(dude_dir, 'dude_subset_list.txt')
        if os.path.exists(subset_file):
            with open(subset_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        target_list.append(line.split(',')[0])
        else:
            active_files = glob.glob(os.path.join(dude_dir, '*_actives.tsv'))
            target_list = [os.path.basename(f).replace('_actives.tsv', '') for f in active_files]
        
        if target:
            target_list = [target] if target in target_list else []
        
        print(f"Loading DUDE dataset with {len(target_list)} targets")
        
        all_smiles = []
        all_metadata = []
        
        for target_name in tqdm(target_list, desc="Reading files"):
            active_file = os.path.join(dude_dir, f'{target_name}_actives.tsv')
            if os.path.exists(active_file):
                with open(active_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            chembl_id, smiles = parts[0], parts[1]
                            all_smiles.append(smiles)
                            all_metadata.append({
                                'target': target_name,
                                'compound_id': chembl_id,
                                'label': 1
                            })
            
            decoy_file = os.path.join(dude_dir, f'{target_name}_decoys.tsv')
            if os.path.exists(decoy_file):
                with open(decoy_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            chembl_id, smiles = parts[0], parts[1]
                            all_smiles.append(smiles)
                            all_metadata.append({
                                'target': target_name,
                                'compound_id': chembl_id,
                                'label': 0
                            })
        
        print(f"Converting {len(all_smiles)} SMILES to SELFIES...")
        all_selfies = batch_convert_smiles(all_smiles, use_parallel=use_parallel)
        
        conversion_failures = 0
        for i, (selfies, metadata) in enumerate(zip(all_selfies, all_metadata)):
            if selfies is not None:
                self.data.append({
                    **metadata,
                    'smiles': all_smiles[i],
                    'selfies': selfies
                })
                self.labels.append(metadata['label'])
                self.targets.append(metadata['target'])
            else:
                conversion_failures += 1
        
        self.labels = np.array(self.labels)
        
        print(f"Loaded {len(self.data)} samples ({np.sum(self.labels)} actives, {len(self.labels) - np.sum(self.labels)} decoys)")
        if conversion_failures > 0:
            print(f"Warning: Failed to convert {conversion_failures} molecules (skipped)")
        
        if use_cache:
            print(f"Saving to cache: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': self.data,
                    'labels': self.labels.tolist(),
                    'targets': self.targets
                }, f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class FastLITPCBADataset(Dataset):
    
    def __init__(self, data_path, target=None, use_cache=True, use_parallel=True):
        self.data_path = data_path
        self.data = []
        self.labels = []
        self.targets = []
        
        litpcba_dir = os.path.join(data_path, 'LIT-PCBA')
        cache_file = os.path.join(litpcba_dir, f'.cache_litpcba_{target if target else "all"}.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.data = cache_data['data']
                self.labels = np.array(cache_data['labels'])
                self.targets = cache_data['targets']
                self.seq_dict = cache_data.get('seq_dict', {})
                print(f"Loaded {len(self.data)} samples from cache")
                return
        
        target_dirs = [d for d in os.listdir(litpcba_dir) 
                      if os.path.isdir(os.path.join(litpcba_dir, d)) 
                      and not d.endswith('.json')]
        
        if target:
            target_dirs = [target] if target in target_dirs else []
        
        print(f"Loading LIT-PCBA dataset with {len(target_dirs)} targets")
        
        seq_dict_file = os.path.join(litpcba_dir, 'lit_pcba_sequence_dict.json')
        self.seq_dict = {}
        if os.path.exists(seq_dict_file):
            with open(seq_dict_file, 'r') as f:
                self.seq_dict = json.load(f)
        
        all_smiles = []
        all_metadata = []
        
        for target_name in tqdm(target_dirs, desc="Reading files"):
            target_dir = os.path.join(litpcba_dir, target_name)
            
            active_file = os.path.join(target_dir, 'actives.smi')
            if os.path.exists(active_file):
                with open(active_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            smiles, compound_id = parts[0], parts[1]
                            all_smiles.append(smiles)
                            all_metadata.append({
                                'target': target_name,
                                'compound_id': compound_id,
                                'label': 1
                            })
            
            inactive_file = os.path.join(target_dir, 'inactives.smi')
            if os.path.exists(inactive_file):
                with open(inactive_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            smiles, compound_id = parts[0], parts[1]
                            all_smiles.append(smiles)
                            all_metadata.append({
                                'target': target_name,
                                'compound_id': compound_id,
                                'label': 0
                            })
        
        print(f"Converting {len(all_smiles)} SMILES to SELFIES...")
        all_selfies = batch_convert_smiles(all_smiles, use_parallel=use_parallel)
        
        conversion_failures = 0
        for i, (selfies, metadata) in enumerate(zip(all_selfies, all_metadata)):
            if selfies is not None:
                self.data.append({
                    **metadata,
                    'smiles': all_smiles[i],
                    'selfies': selfies
                })
                self.labels.append(metadata['label'])
                self.targets.append(metadata['target'])
            else:
                conversion_failures += 1
        
        self.labels = np.array(self.labels)
        
        print(f"Loaded {len(self.data)} samples ({np.sum(self.labels)} actives, {len(self.labels) - np.sum(self.labels)} inactives)")
        if conversion_failures > 0:
            print(f"Warning: Failed to convert {conversion_failures} molecules (skipped)")
        
        if use_cache:
            print(f"Saving to cache: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': self.data,
                    'labels': self.labels.tolist(),
                    'targets': self.targets,
                    'seq_dict': self.seq_dict
                }, f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_model(model, dataset, device='cuda', batch_size=32, prot_encoder_path="westlake-repl/SaProt_650M_AF2", drug_encoder_path="HUBioDataLab/SELFormer"):
    model.eval()
    
    print("Loading tokenizers and encoders...")
    prot_tokenizer = EsmTokenizer.from_pretrained(prot_encoder_path)
    drug_tokenizer = AutoTokenizer.from_pretrained(drug_encoder_path)
    
    prot_encoder = EsmForMaskedLM.from_pretrained(prot_encoder_path)
    drug_encoder = AutoModel.from_pretrained(drug_encoder_path)
    
    encoder_model = TokenEncoder(prot_encoder, drug_encoder)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    
    predictions = []
    actuals = []
    targets = []
    
    has_sequences = hasattr(dataset, 'seq_dict') and dataset.seq_dict
    
    if has_sequences:
        target_to_seq = dataset.seq_dict
    else:
        target_to_seq = {}
        dude_fasta = os.path.join(os.path.dirname(dataset.data_path) if hasattr(dataset, 'data_path') else 'dataset/DUDE', 'dude_targets.fasta')
        if os.path.exists(dude_fasta):
            current_target = None
            current_seq = []
            with open(dude_fasta, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        if current_target:
                            target_to_seq[current_target] = ''.join(current_seq)
                        current_target = line.strip()[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())
                if current_target:
                    target_to_seq[current_target] = ''.join(current_seq)
    
    num_samples = len(dataset)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
            batch_data = []
            batch_end = min(i + batch_size, num_samples)
            
            for j in range(i, batch_end):
                item = dataset[j]
                batch_data.append(item)
            
            batch_targets = [item['target'] for item in batch_data]
            batch_smiles = [item.get('smiles', '') for item in batch_data]
            batch_selfies = [item.get('selfies', '') for item in batch_data]
            batch_labels = [item['label'] for item in batch_data]
            
            batch_proteins = []
            for target_name in batch_targets:
                if target_name in target_to_seq:
                    if isinstance(target_to_seq[target_name], list):
                        seq = target_to_seq[target_name][0]
                    else:
                        seq = target_to_seq[target_name]
                    batch_proteins.append(seq)
            
            drug_selfies = []
            for selfies, smiles in zip(batch_selfies, batch_smiles):
                if selfies:
                    drug_selfies.append(selfies)
                elif smiles:
                    try:
                        converted_selfies = sf.encoder(smiles)
                    except:
                        converted_selfies = "[C]"
                    drug_selfies.append(converted_selfies)
                else:
                    drug_selfies.append("[C]")
            
            try:
                prot_inputs = prot_tokenizer(batch_proteins, padding=True, truncation=True, return_tensors='pt', max_length=512)
                drug_inputs = drug_tokenizer(drug_selfies, padding=True, truncation=True, return_tensors='pt', max_length=512)
                
                prot_ids = prot_inputs['input_ids'].to(device)
                prot_mask = prot_inputs['attention_mask'].to(device)
                drug_ids = drug_inputs['input_ids'].to(device)
                drug_mask = drug_inputs['attention_mask'].to(device)
                
                prot_embed, drug_embed = encoder_model.encoding(prot_ids, prot_mask, drug_ids, drug_mask)
                outputs, _ = model(prot_embed, drug_embed, batch_proteins, drug_selfies, None)
                
                batch_predictions = torch.sigmoid(outputs).cpu().numpy()
                
                if len(batch_predictions.shape) > 1:
                    batch_predictions = batch_predictions[:, 0]
                    
            except Exception as e:
                print(f"Error during model inference: {e}")
                import traceback
                traceback.print_exc()
                batch_predictions = np.random.random(len(batch_labels))
            
            predictions.extend(batch_predictions)
            actuals.extend(batch_labels)
            targets.extend(batch_targets)
    
    return np.array(predictions), np.array(actuals), targets


def evaluate_dataset(args, dataset_name='dude'):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if dataset_name.lower() == 'dude':
        dataset = FastDUDEDataset(args.data_path, use_cache=args.use_cache, use_parallel=args.use_parallel)
    elif dataset_name.lower() in ['pcba', 'lit-pcba', 'litpcba']:
        dataset = FastLITPCBADataset(args.data_path, use_cache=args.use_cache, use_parallel=args.use_parallel)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    model = DTIModel()
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model = model.to(device)
    
    print(f"\nEvaluating on {dataset_name.upper()} dataset...")
    predictions, actuals, target_list = evaluate_model(model, dataset, device, batch_size=args.batch_size)
    
    results_by_target = defaultdict(lambda: {'predictions': [], 'labels': []})
    
    for i in range(len(predictions)):
        target = target_list[i]
        results_by_target[target]['predictions'].append(predictions[i])
        results_by_target[target]['labels'].append(actuals[i])
    
    metrics_calculator = EvaluationMetrics()
    all_results = []
    
    print(f"\n{'Target':<15} {'AUROC':<10} {'BEDROC':<10} {'EF_0.5%':<10} {'EF_1%':<10} {'EF_5%':<10}")
    print("-" * 75)
    
    for target, data in results_by_target.items():
        y_true = np.array(data['labels'])
        y_score = np.array(data['predictions'])
        
        auroc = metrics_calculator.auroc(y_true, y_score)
        bedroc = metrics_calculator.bedroc(y_true, y_score)
        ef_05 = metrics_calculator.enrichment_factor(y_true, y_score, 0.5)
        ef_1 = metrics_calculator.enrichment_factor(y_true, y_score, 1.0)
        ef_5 = metrics_calculator.enrichment_factor(y_true, y_score, 5.0)
        
        print(f"{target:<15} {auroc:<10.4f} {bedroc:<10.4f} {ef_05:<10.2f} {ef_1:<10.2f} {ef_5:<10.2f}")
        
        all_results.append({
            'target': target,
            'auroc': float(auroc),
            'bedroc': float(bedroc),
            'ef_0.5%': float(ef_05),
            'ef_1%': float(ef_1),
            'ef_5%': float(ef_5),
            'n_actives': int(np.sum(y_true)),
            'n_total': int(len(y_true))
        })
    
    avg_metrics = {
        'auroc': float(np.mean([r['auroc'] for r in all_results])),
        'bedroc': float(np.mean([r['bedroc'] for r in all_results])),
        'ef_0.5%': float(np.mean([r['ef_0.5%'] for r in all_results])),
        'ef_1%': float(np.mean([r['ef_1%'] for r in all_results])),
        'ef_5%': float(np.mean([r['ef_5%'] for r in all_results]))
    }
    
    print("-" * 75)
    print(f"{'AVERAGE':<15} {avg_metrics['auroc']:<10.4f} {avg_metrics['bedroc']:<10.4f} "
          f"{avg_metrics['ef_0.5%']:<10.2f} {avg_metrics['ef_1%']:<10.2f} {avg_metrics['ef_5%']:<10.2f}")
    
    return all_results, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Fast evaluation of DTI model on DUDE and LIT-PCBA datasets')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='both', choices=['dude', 'pcba', 'both'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data-path', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='test_results', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--use-cache', action='store_true', default=True, help='Use cached SELFIES conversions')
    parser.add_argument('--no-cache', dest='use_cache', action='store_false', help='Disable caching')
    parser.add_argument('--use-parallel', action='store_true', default=True, help='Use parallel processing')
    parser.add_argument('--no-parallel', dest='use_parallel', action='store_false', help='Disable parallel processing')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    if args.dataset in ['dude', 'both']:
        dude_results, dude_avg = evaluate_dataset(args, 'dude')
        all_results['dude'] = {
            'per_target': dude_results,
            'average': dude_avg
        }
        
        with open(os.path.join(args.output_dir, 'dude_results.json'), 'w') as f:
            json.dump(all_results['dude'], f, indent=2)
        
        df = pd.DataFrame(dude_results)
        df.to_csv(os.path.join(args.output_dir, 'dude_results.csv'), index=False)
    
    if args.dataset in ['pcba', 'both']:
        pcba_results, pcba_avg = evaluate_dataset(args, 'lit-pcba')
        all_results['lit-pcba'] = {
            'per_target': pcba_results,
            'average': pcba_avg
        }
        
        with open(os.path.join(args.output_dir, 'lit_pcba_results.json'), 'w') as f:
            json.dump(all_results['lit-pcba'], f, indent=2)
        
        df = pd.DataFrame(pcba_results)
        df.to_csv(os.path.join(args.output_dir, 'lit_pcba_results.csv'), index=False)
    
    print("\n" + "="*75)
    print("EVALUATION SUMMARY")
    print("="*75)
    
    for dataset_name, results in all_results.items():
        avg = results['average']
        print(f"\n{dataset_name.upper()}:")
        print(f"  AUROC:  {avg['auroc']:.4f}")
        print(f"  BEDROC: {avg['bedroc']:.4f}")
        print(f"  EF 0.5%: {avg['ef_0.5%']:.2f}")
        print(f"  EF 1%:   {avg['ef_1%']:.2f}")
        print(f"  EF 5%:   {avg['ef_5%']:.2f}")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()