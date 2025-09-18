import argparse
import os
import sys
from datetime import datetime
import gc
import warnings

warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append("../")
import numpy as np
import swanlab
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from transformers import EsmTokenizer, EsmForMaskedLM, AutoModel, AutoTokenizer

# Import from local modules
from dataset import DatasetProcessor, BatchFileDataset, load_dataset
from model import DTIModel, TokenEncoder
import selfies

# Use environment variable for API key or use default
try:
    api_key = os.environ.get('SWANLAB_API_KEY', 'BXKiHFxqexqRyWelXBEnH')
    swanlab.login(api_key=api_key)
except Exception as e:
    print(f"Warning: Could not login to SwanLab: {e}")
    pass

def get_dataset_name(args):
    return os.path.basename(args.data_path).replace('.csv', '')

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument(
        "--prot_encoder_path",
        type=str,
        default="westlake-repl/SaProt_650M_AF2",
        # westlake-repl/SaProt_650M_PDB
        help="path/name of protein encoder model located",
    )
    parser.add_argument(
        "--drug_encoder_path",
        type=str,
        default="HUBioDataLab/SELFormer",
        # "ibm/MoLFormer-XL-both-10pct"
        help="path/name of SMILE pre-trained language model",
    )
    parser.add_argument(
        "--token_cache",
        type=str,
        default="dataset/processed_token",
        help="path of tokenized training data",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--use_pooled", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoint/",
        help="save the result in which directory",
    )
    parser.add_argument(
        "--save_name", default="fine_tune", type=str, help="the name of the saved file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data CSV file (e.g., /path/to/BindingDB.csv)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=500,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for cross validation"
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Use stratified k-fold splitting"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    parser.add_argument(
        "--exclusion_file",
        type=str,
        default=None,
        help="File with protein IDs to exclude (for homology-based analysis in MERGED dataset)"
    )
    parser.add_argument(
        "--train_only_excluded",
        action="store_true",
        help="Train only on proteins in the exclusion file (inverts exclusion logic)"
    )
    parser.add_argument(
        "--neg_sample_ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive samples (for MERGED dataset)"
    )
    parser.add_argument(
        "--protein_filter_file",
        type=str,
        default=None,
        help="File containing proteins to filter from DUDE or PCBA datasets"
    )
    parser.add_argument(
        "--protein_filter_type",
        type=str,
        choices=["DUDE", "PCBA"],
        default=None,
        help="Type of protein filter file (DUDE or PCBA)"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="K-mer Jaccard similarity threshold for filtering proteins (default: 0.9)"
    )
    return parser.parse_args()

def collate_fn_batch_encoding(batch, prot_tokenizer, drug_tokenizer):
    if len(batch[0]) == 5:
        prot_seqs, drug_seqs, scores, raw_prot, raw_smiles = zip(*batch)
    elif len(batch[0]) == 3:
        prot_seqs, drug_seqs, scores = zip(*batch)
        raw_prot = prot_seqs
        raw_smiles = drug_seqs
    else:
        raise ValueError(f"Unexpected batch format with {len(batch[0])} elements")
    
    converted_smiles = []
    for drug in raw_smiles:
        try:
            if drug.startswith('[') and drug.endswith(']') and '.' not in drug:
                smiles = selfies.decoder(drug)
                converted_smiles.append(smiles)
            else:
                converted_smiles.append(drug)
        except:
            converted_smiles.append(drug)
    raw_smiles = converted_smiles
    
    prot_enc = prot_tokenizer.batch_encode_plus(
        list(prot_seqs), max_length=512, padding="max_length", truncation=True,
        add_special_tokens=True, return_tensors="pt"
    )
    drug_enc = drug_tokenizer.batch_encode_plus(
        list(drug_seqs), max_length=512, padding="max_length", truncation=True,
        add_special_tokens=True, return_tensors="pt"
    )
    scores = torch.tensor(list(scores))
    
    return (prot_enc["input_ids"], prot_enc["attention_mask"].bool(), 
            drug_enc["input_ids"], drug_enc["attention_mask"].bool(), 
            scores, list(raw_prot), list(raw_smiles))

def get_feature(model, dataloader, args, set_type):
    dataset_name = get_dataset_name(args)
    subdirectory = os.path.join(args.token_cache, dataset_name)
    os.makedirs(subdirectory, exist_ok=True)
    
    batch_files = []
    batch_number = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader)):
            if len(batch) == 7:
                prot_ids, prot_mask, drug_ids, drug_mask, labels, prot_seqs, drug_smiles = batch
            else:
                prot_ids, prot_mask, drug_ids, drug_mask, labels = batch
                prot_seqs = None
                drug_smiles = None
                
            prot_ids = prot_ids.to(args.device)
            prot_mask = prot_mask.to(args.device)
            drug_ids = drug_ids.to(args.device)
            drug_mask = drug_mask.to(args.device)
            
            prot_embed, drug_embed = model.encoding(prot_ids, prot_mask, drug_ids, drug_mask)
            prot_embed = prot_embed.cpu()
            drug_embed = drug_embed.cpu()
            prot_mask = prot_mask.cpu()
            drug_mask = drug_mask.cpu()
            labels = labels.cpu()
            
            # Clear GPU memory
            del prot_ids, drug_ids
            torch.cuda.empty_cache()
        
            # Save each batch to a separate file in the subdirectory
            batch_file = os.path.join(
                subdirectory,
                f"{dataset_name}_{set_type}_batch_{batch_number}.pt"
            )
            # Save embeddings and sequences
            save_dict = {
                'prot': prot_embed,
                'drug': drug_embed,
                'prot_mask': prot_mask,
                'drug_mask': drug_mask,
                'y': labels
            }
            
            if prot_seqs is not None:
                save_dict['prot_seqs'] = prot_seqs
            if drug_smiles is not None:
                save_dict['drug_smiles'] = drug_smiles
                
            torch.save(save_dict, batch_file)
            batch_files.append(batch_file)
            batch_number += 1
    return batch_files

def get_data_loader(file_list, batch_size, shuffle=False, num_workers=0):
    dataset = BatchFileDataset(file_list)
    # Use num_workers=0 to avoid multiprocessing memory issues
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x[0], pin_memory=False)

def encode_and_save_features(args, prot_tokenizer, drug_tokenizer, dataset, output_dir, prefix, encoder_model=None):
    """Encode and save features for a dataset split"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataloader
    def collate_fn(batch):
        return collate_fn_batch_encoding(batch, prot_tokenizer, drug_tokenizer)
    
    encoding_batch_size = args.batch_size
    if prefix.startswith("MERGED"):
        try:
            if torch.cuda.is_available():
                # Get GPU memory
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                encoding_batch_size = args.batch_size
                print(f"  Using encoding batch size: {encoding_batch_size} (GPU memory: {gpu_mem:.1f}GB)")
        except:
            pass
    
    dataloader = DataLoader(
        dataset, batch_size=encoding_batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=4, pin_memory=True  # Enable pin_memory and multiple workers for faster data loading
    )
    
    # Use provided model or create new one
    if encoder_model is None:
        prot_model = EsmForMaskedLM.from_pretrained(args.prot_encoder_path)
        drug_model = AutoModel.from_pretrained(args.drug_encoder_path)
        model = TokenEncoder(prot_model, drug_model)
        model = model.to(args.device)
        should_cleanup = True
    else:
        model = encoder_model
        should_cleanup = False
    
    batch_files = []
    batch_number = 0
    
    # Enable automatic mixed precision for faster encoding
    use_amp = torch.cuda.is_available() and prefix.startswith("MERGED")
    if use_amp:
        from torch.cuda.amp import autocast
        print(f"  Using automatic mixed precision (AMP) for faster encoding")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {prefix}", ncols=100):
            if len(batch) == 7:
                prot_ids, prot_mask, drug_ids, drug_mask, labels, prot_seqs, drug_smiles = batch
            else:
                prot_ids, prot_mask, drug_ids, drug_mask, labels = batch
                prot_seqs = None
                drug_smiles = None
            
            # Move to GPU with non-blocking transfer
            prot_ids = prot_ids.to(args.device, non_blocking=True)
            prot_mask = prot_mask.to(args.device, non_blocking=True)
            drug_ids = drug_ids.to(args.device, non_blocking=True)
            drug_mask = drug_mask.to(args.device, non_blocking=True)
            
            # Encode with optional mixed precision
            if use_amp:
                with autocast():
                    prot_embed, drug_embed = model.encoding(prot_ids, prot_mask, drug_ids, drug_mask)
                    # Convert back to float32 for storage
                    prot_embed = prot_embed.float()
                    drug_embed = drug_embed.float()
            else:
                prot_embed, drug_embed = model.encoding(prot_ids, prot_mask, drug_ids, drug_mask)
            
            # Move to CPU with non-blocking transfer
            prot_embed = prot_embed.cpu()
            drug_embed = drug_embed.cpu()
            prot_mask = prot_mask.cpu()
            drug_mask = drug_mask.cpu()
            labels = labels.cpu()
            
            # Clear GPU memory less frequently for better performance
            if batch_number % 10 == 0:
                del prot_ids, drug_ids
                torch.cuda.empty_cache()
            else:
                del prot_ids, drug_ids
            
            # Save batch
            batch_file = os.path.join(output_dir, f"{prefix}_batch_{batch_number}.pt")
            save_dict = {
                'prot': prot_embed,
                'drug': drug_embed,
                'prot_mask': prot_mask,
                'drug_mask': drug_mask,
                'y': labels
            }
            
            if prot_seqs is not None and len(prot_seqs) > 0:
                save_dict['prot_seqs'] = prot_seqs
            else:
                print(f"WARNING: No protein sequences to save for batch {batch_number}")
            
            if drug_smiles is not None and len(drug_smiles) > 0:
                save_dict['drug_smiles'] = drug_smiles
            else:
                print(f"WARNING: No drug SMILES to save for batch {batch_number}")
            
            torch.save(save_dict, batch_file)
            batch_files.append(batch_file)
            batch_number += 1
    
    # Clean up models only if we created them
    if should_cleanup:
        del model
        if 'prot_model' in locals():
            del prot_model, drug_model
        torch.cuda.empty_cache()
        gc.collect()
    
    return batch_files

def encode_pretrained_feature(args, prot_tokenizer, drug_tokenizer):
    dataset_name = get_dataset_name(args)
    input_feat_path = os.path.join(args.token_cache, dataset_name)
    os.makedirs(input_feat_path, exist_ok=True)
    train_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if f.startswith(f"{dataset_name}_train_batch")])
    valid_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if f.startswith(f"{dataset_name}_valid_batch")])
    test_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if f.startswith(f"{dataset_name}_test_batch")])
    
    if train_files and valid_files and test_files:
        print("Batch files found and will be used.")
    else:
        print("Creating pre-encoded features...")
        prot_model = EsmForMaskedLM.from_pretrained(args.prot_encoder_path)
        drug_model = AutoModel.from_pretrained(args.drug_encoder_path)
        
        model = TokenEncoder(prot_model, drug_model)
        model = model.to(args.device)
            
        def collate_fn(batch):
            return collate_fn_batch_encoding(batch, prot_tokenizer, drug_tokenizer)
            
        Dataset = load_dataset(args)
        train_examples = Dataset.get_train_examples()
        valid_examples = Dataset.get_val_examples()
        test_examples = Dataset.get_test_examples()

        train_dataloader = DataLoader(
            train_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        valid_dataloader = DataLoader(
            valid_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        test_dataloader = DataLoader(
            test_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        print(f"Dataset loaded: train-{len(train_examples)}; valid-{len(valid_examples)}; test-{len(test_examples)}")

        train_files = get_feature(model, train_dataloader, args, "train")
        valid_files = get_feature(model, valid_dataloader, args, "valid")
        test_files = get_feature(model, test_dataloader, args, "test")
        
        # Explicitly delete models to free memory
        del model, prot_model, drug_model
        torch.cuda.empty_cache()
        gc.collect()

    return train_files, valid_files, test_files

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, best_model_dir, num_epochs=200, patience=10):
    best_auc = 0
    best_model = None
    epochs = 0  # Initialize counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_cl_loss = 0
        
        # Training phase
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            if len(batch) == 7:
                # Batch includes sequences for similarity calculations
                prot, drug, prot_mask, drug_mask, labels, prot_seqs, drug_smiles = batch
                prot, drug, prot_mask, drug_mask, labels = (
                    prot.to(device), 
                    drug.to(device),
                    prot_mask.to(device),
                    drug_mask.to(device), 
                    labels.to(device)
                )
                optimizer.zero_grad()
                output, cl_loss = model(prot, drug, prot_seqs, drug_smiles, labels)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements. Expected 5 or 7 elements.")
                
            # Combine losses
            bce_loss = criterion(output, labels.unsqueeze(1).float())
            loss = bce_loss + cl_loss
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at batch {batch_idx}, epoch {epoch+1}")
                print(f"BCE Loss: {bce_loss.item()}, CL Loss: {cl_loss.item() if isinstance(cl_loss, torch.Tensor) else cl_loss}")
                optimizer.zero_grad()
                continue
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += bce_loss.item()
            total_cl_loss += cl_loss.item() if isinstance(cl_loss, torch.Tensor) else cl_loss
            
            # Clear intermediate variables to free memory
            del output, bce_loss, loss, cl_loss
            if 'prot' in locals():
                del prot, drug, prot_mask, drug_mask
            if 'prot_ids' in locals():
                del prot_ids, drug_ids
            del labels
            
            # Periodic memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        scheduler.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            predictions, actuals = [], []
            for batch in valid_loader:
                if len(batch) == 7:
                    # Batch includes sequences
                    prot, drug, prot_mask, drug_mask, labels, prot_seqs, drug_smiles = batch
                    prot, drug, prot_mask, drug_mask, labels = (
                        prot.to(device), 
                        drug.to(device), 
                        prot_mask.to(device), 
                        drug_mask.to(device), 
                        labels.to(device)
                    )
                    output, _ = model(prot, drug, prot_seqs, drug_smiles, labels)
                elif len(batch) == 5:
                    # Legacy format without sequences
                    prot, drug, prot_mask, drug_mask, labels = batch
                    prot, drug, prot_mask, drug_mask, labels = (
                        prot.to(device), 
                        drug.to(device), 
                        prot_mask.to(device), 
                        drug_mask.to(device), 
                        labels.to(device)
                    )
                    output, _ = model(prot, drug, None, None, labels)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements. Expected 5 or 7 elements.")
                    
                probs = output
                predictions.extend(probs.squeeze().cpu().numpy())
                actuals.extend(labels.cpu().numpy())
                
                # Clean up validation batch
                del output
                if 'prot' in locals():
                    del prot, drug, prot_mask, drug_mask
                if 'prot_input' in locals():
                    del prot_input, drug_input
                del labels
                
            auc = roc_auc_score(actuals, predictions)
            print(f'Epoch {epoch+1}: Validation AUC: {auc:.4f}, BCE Loss: {total_loss / len(train_loader):.4f}, CL Loss: {total_cl_loss / len(train_loader):.4f}')
            
            # Log metrics to swanlab
            swanlab.log({
                "epoch": epoch + 1, 
                "bce_loss": total_loss / len(train_loader), 
                "cl_loss": total_cl_loss / len(train_loader),
                "val_auc": auc
            })

            if auc > best_auc:
                best_auc = auc
                best_model = model.state_dict()
                # Save the best model
                torch.save(best_model, f'{best_model_dir}/best_model.ckpt')
                epochs = 0
            else:
                epochs += 1

            if epochs >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
        gc.collect()

    return best_model

def train_kfold(model_class, dataset_processor, prot_tokenizer, drug_tokenizer, args, device):
    fold_results = []
    all_test_predictions = []
    all_test_actuals = []
    
    print(f"\n{'='*60}\nK-Fold Cross Validation Configuration\n{'='*60}")
    
    fold_info = dataset_processor.get_fold_info()
    
    # Handle both regular and MERGED dataset processors
    if hasattr(dataset_processor, 'csv_file'):
        dataset_name = os.path.basename(dataset_processor.csv_file).replace('.csv', '')
    else:
        dataset_name = getattr(dataset_processor, 'name', 'Unknown')
    
    print(f"\nDataset: {dataset_name}")
    print(f"Total Folds: {dataset_processor.get_num_folds()}")
    
    if hasattr(dataset_processor, 'config_file'):
        print(f"Configuration file: {dataset_processor.config_file}")
    print("\nFold Summary:")
    
    for info in fold_info:
        print(f"  Fold {info['fold'] + 1}: Train={info['train_size']:4}, Valid={info['valid_size']:3}, "
              f"Test={info['test_size']:3}")
    
    features_exist_count = sum(1 for info in fold_info if info["features_exist"])
    if features_exist_count:
        print(f"\n{features_exist_count}/{len(fold_info)} folds have pre-encoded features available.")
    
    # Pre-create encoder models once if needed (check all folds first)
    encoder_models_needed = False
    for fold in range(dataset_processor.get_num_folds()):
        dataset_processor.set_fold(fold)
        fold_cache_dir = os.path.join(args.token_cache, dataset_processor.name, f"fold_{fold + 1}")
        if not os.path.exists(fold_cache_dir):
            encoder_models_needed = True
            break
        else:
            files = os.listdir(fold_cache_dir)
            train_files = [f for f in files if f.startswith(f"{dataset_processor.name}_train_batch")]
            valid_files = [f for f in files if f.startswith(f"{dataset_processor.name}_valid_batch")]
            test_files = [f for f in files if f.startswith(f"{dataset_processor.name}_test_batch")]
            if not (train_files and valid_files and test_files):
                encoder_models_needed = True
                break
    
    # Load encoder models once if needed
    prot_model = None
    drug_model = None
    if encoder_models_needed:
        print("Loading encoder models for feature extraction...")
        prot_model = EsmForMaskedLM.from_pretrained(args.prot_encoder_path)
        drug_model = AutoModel.from_pretrained(args.drug_encoder_path)
    
    for fold in range(dataset_processor.get_num_folds()):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/{dataset_processor.get_num_folds()}")
        print(f"{'='*50}")
        
        # Set the current fold
        dataset_processor.set_fold(fold)
        
        # Initialize model for this fold
        model = model_class(
            prot_dim=1280,
            drug_dim=768,
            latent_dim=1024,
            num_heads=8,
            dropout=args.dropout
        ).to(device)
        
        # Setup training components
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)
        
        # Create fold-specific save directory
        fold_save_dir = os.path.join(args.save_name, f"fold_{fold + 1}")
        os.makedirs(fold_save_dir, exist_ok=True)
        
        # Setup data loaders based on mode
        # Pre-encode features for this fold
        # Clean up any previous fold's data before starting new fold
        if fold > 0:
            print(f"Cleaning up memory from previous fold...")
            gc.collect()
            torch.cuda.empty_cache()
        
        # Set the current fold in dataset processor
        dataset_processor.set_fold(fold)
        
        # Check if pre-encoded features exist, otherwise encode them
        fold_cache_dir = os.path.join(args.token_cache, dataset_processor.name, f"fold_{fold + 1}")
        
        if os.path.exists(fold_cache_dir):
            files = os.listdir(fold_cache_dir)
            train_files = sorted([os.path.join(fold_cache_dir, f) for f in files if f.startswith(f"{dataset_processor.name}_train_batch")])
            valid_files = sorted([os.path.join(fold_cache_dir, f) for f in files if f.startswith(f"{dataset_processor.name}_valid_batch")])
            test_files = sorted([os.path.join(fold_cache_dir, f) for f in files if f.startswith(f"{dataset_processor.name}_test_batch")])
            
            if train_files and valid_files and test_files:
                # Check if the files have sequences using memory mapping
                sample_file = torch.load(train_files[0], map_location='cpu') if train_files else None
                has_sequences = sample_file and 'prot_seqs' in sample_file and 'drug_smiles' in sample_file
                del sample_file  # Free memory immediately
                
                if has_sequences:
                    print(f"Using existing pre-encoded features for fold {fold + 1} (with sequences)")
                else:
                    print(f"Existing features lack sequences, regenerating for fold {fold + 1}")
                    train_files = valid_files = test_files = None  # Force regeneration
            else:
                train_files = valid_files = test_files = None
            
            if not (train_files and valid_files and test_files):
                # Need to encode features for this fold
                print(f"Creating pre-encoded features for fold {fold + 1}...")
                
                # Use pre-loaded models if available
                if prot_model is not None and drug_model is not None:
                    encoder_model = TokenEncoder(prot_model, drug_model).to(args.device)
                else:
                    # Load models if not pre-loaded
                    temp_prot_model = EsmForMaskedLM.from_pretrained(args.prot_encoder_path)
                    temp_drug_model = AutoModel.from_pretrained(args.drug_encoder_path)
                    encoder_model = TokenEncoder(temp_prot_model, temp_drug_model).to(args.device)
                
                train_files = encode_and_save_features(
                    args, prot_tokenizer, drug_tokenizer, 
                    dataset_processor.get_train_examples(), 
                    fold_cache_dir, f"{dataset_processor.name}_train",
                    encoder_model=encoder_model
                )
                valid_files = encode_and_save_features(
                    args, prot_tokenizer, drug_tokenizer,
                    dataset_processor.get_val_examples(),
                    fold_cache_dir, f"{dataset_processor.name}_valid",
                    encoder_model=encoder_model
                )
                test_files = encode_and_save_features(
                    args, prot_tokenizer, drug_tokenizer,
                    dataset_processor.get_test_examples(),
                    fold_cache_dir, f"{dataset_processor.name}_test",
                    encoder_model=encoder_model
                )
                
                # Clean up encoder model
                del encoder_model
                if 'temp_prot_model' in locals():
                    del temp_prot_model, temp_drug_model
                torch.cuda.empty_cache()
                gc.collect()
        else:
            # Need to encode features for this fold
            print(f"Creating pre-encoded features for fold {fold + 1}...")
            
            # Use pre-loaded models if available
            if prot_model is not None and drug_model is not None:
                encoder_model = TokenEncoder(prot_model, drug_model).to(args.device)
            else:
                # Load models if not pre-loaded
                temp_prot_model = EsmForMaskedLM.from_pretrained(args.prot_encoder_path)
                temp_drug_model = AutoModel.from_pretrained(args.drug_encoder_path)
                encoder_model = TokenEncoder(temp_prot_model, temp_drug_model).to(args.device)
            
            # Create data loaders with collate function
            def collate_fn(batch):
                return collate_fn_batch_encoding(batch, prot_tokenizer, drug_tokenizer)
            
            train_dataloader = DataLoader(
                dataset_processor.get_train_examples(),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            valid_dataloader = DataLoader(
                dataset_processor.get_val_examples(),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            test_dataloader = DataLoader(
                dataset_processor.get_test_examples(),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            
            # Get pre-encoded features
            train_files = get_feature_fold(encoder_model, train_dataloader, args, "train", fold + 1)
            valid_files = get_feature_fold(encoder_model, valid_dataloader, args, "valid", fold + 1)
            test_files = get_feature_fold(encoder_model, test_dataloader, args, "test", fold + 1)
            
            # Clean up encoder models
            del encoder_model
            if 'temp_prot_model' in locals():
                del temp_prot_model, temp_drug_model
            torch.cuda.empty_cache()
            gc.collect()
            
            # Update feature status
            dataset_processor.update_feature_status(fold, saved=True)
        train_loader = get_data_loader(train_files, batch_size=1, shuffle=True)
        valid_loader = get_data_loader(valid_files, batch_size=1, shuffle=False)
        test_loader = get_data_loader(test_files, batch_size=1, shuffle=False)
        
        # Train model for this fold
        print(f"Training fold {fold + 1}...")
        best_model = train(
            model, train_loader, valid_loader, criterion, optimizer, scheduler,
            device, fold_save_dir, num_epochs=args.num_epochs, patience=args.patience
        )
        
        # Test model for this fold
        model.load_state_dict(best_model)
        fold_auc, fold_aupr, fold_accuracy, fold_predictions, fold_actuals = test_fold(model, test_loader, device)
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'test_auc': fold_auc,
            'test_aupr': fold_aupr,
            'test_accuracy': fold_accuracy
        }
        fold_results.append(fold_result)
        
        # Accumulate predictions for overall evaluation
        all_test_predictions.extend(fold_predictions)
        all_test_actuals.extend(fold_actuals)
        
        # Log fold results
        print(f"Fold {fold + 1} Results: AUC={fold_auc:.4f}, AUPR={fold_aupr:.4f}, Accuracy={fold_accuracy:.4f}")
        swanlab.log({
            f"fold_{fold + 1}_test_auc": fold_auc,
            f"fold_{fold + 1}_test_aupr": fold_aupr,
            f"fold_{fold + 1}_test_accuracy": fold_accuracy,
        })
        
        # Clean up fold resources
        del model, optimizer, scheduler
        # Clean up data loaders explicitly
        del train_loader, valid_loader, test_loader
        # Clean up any remaining batch files from memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Force Python garbage collection
        print(f"Memory cleanup completed for fold {fold + 1}")
    
    # Clean up pre-loaded encoder models if they exist
    if prot_model is not None:
        del prot_model
    if drug_model is not None:
        del drug_model
    torch.cuda.empty_cache()
    gc.collect()
    
    overall_auc = roc_auc_score(all_test_actuals, all_test_predictions)
    overall_aupr = average_precision_score(all_test_actuals, all_test_predictions)
    overall_accuracy = accuracy_score(all_test_actuals, np.array(all_test_predictions) > 0.5)
    
    aucs = [r['test_auc'] for r in fold_results]
    auprs = [r['test_aupr'] for r in fold_results]
    accuracies = [r['test_accuracy'] for r in fold_results]
    
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    mean_aupr, std_aupr = np.mean(auprs), np.std(auprs)
    mean_accuracy, std_accuracy = np.mean(accuracies), np.std(accuracies)
    
    print(f"\n{'='*60}")
    print("K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Individual Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: AUC={result['test_auc']:.4f}, "
              f"AUPR={result['test_aupr']:.4f}, Accuracy={result['test_accuracy']:.4f}")
    
    print(f"\nCross-Validation Statistics:")
    print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Mean AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}")
    print(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    print(f"\nOverall Results (All Predictions):")
    print(f"  Overall AUC: {overall_auc:.4f}")
    print(f"  Overall AUPR: {overall_aupr:.4f}")
    print(f"  Overall Accuracy: {overall_accuracy:.4f}")
    
    # Log final results to swanlab
    swanlab.log({
        "cv_mean_auc": mean_auc,
        "cv_std_auc": std_auc,
        "cv_mean_aupr": mean_aupr,
        "cv_std_aupr": std_aupr,
        "cv_mean_accuracy": mean_accuracy,
        "cv_std_accuracy": std_accuracy,
        "overall_auc": overall_auc,
        "overall_aupr": overall_aupr,
        "overall_accuracy": overall_accuracy,
    })
    
    return fold_results

def test_fold(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if len(batch) == 7:
                # Batch includes sequences
                prot, drug, prot_mask, drug_mask, labels, prot_seqs, drug_smiles = batch
                prot, drug = prot.to(device), drug.to(device)
                prot_mask, drug_mask = prot_mask.to(device), drug_mask.to(device)
                labels = labels.to(device)
                output, _ = model(prot, drug, prot_seqs, drug_smiles, labels)
            elif len(batch) == 5:
                # Legacy format without sequences
                prot, drug, prot_mask, drug_mask, labels = batch
                prot, drug = prot.to(device), drug.to(device)
                prot_mask, drug_mask = prot_mask.to(device), drug_mask.to(device)
                labels = labels.to(device)
                output, _ = model(prot, drug, prot_mask, drug_mask, labels)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements. Expected 5 or 7 elements.")
                
            # Handle NaN values (output already has sigmoid applied)
            probs = output
            if torch.isnan(probs).any():
                print(f"Warning: NaN detected in predictions during testing")
                # Replace NaN with 0.5 (neutral prediction)
                probs = torch.nan_to_num(probs, nan=0.5)
            predictions.extend(probs.squeeze().cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            
            # Clean up test batch
            del output
            if 'prot' in locals():
                del prot, drug, prot_mask, drug_mask
            if 'prot_ids' in locals():
                del prot_ids, drug_ids
            del labels
    
    # Clear cache after testing
    torch.cuda.empty_cache()
    
    auc = roc_auc_score(actuals, predictions)
    aupr = average_precision_score(actuals, predictions)
    accuracy = accuracy_score(actuals, np.array(predictions) > 0.5)
    
    return auc, aupr, accuracy, predictions, actuals

def encode_pretrained_feature_fold(args, prot_tokenizer, drug_tokenizer, dataset_processor, fold):
    """Encode and save pretrained features for a specific fold"""
    dataset_name = get_dataset_name(args)
    fold_input_path = os.path.join(args.token_cache, dataset_name, f"fold_{fold + 1}")
    os.makedirs(fold_input_path, exist_ok=True)
    
    if os.path.exists(fold_input_path):
        files = os.listdir(fold_input_path)
        train_files = sorted([os.path.join(fold_input_path, f) for f in files if f.startswith(f"{dataset_name}_train_batch")])
        valid_files = sorted([os.path.join(fold_input_path, f) for f in files if f.startswith(f"{dataset_name}_valid_batch")])
        test_files = sorted([os.path.join(fold_input_path, f) for f in files if f.startswith(f"{dataset_name}_test_batch")])
    else:
        train_files = valid_files = test_files = []
    
    if train_files and valid_files and test_files:
        print(f"Pre-encoded features for fold {fold + 1} found and will be reused.")
        print(f"Train batches: {len(train_files)}, Valid batches: {len(valid_files)}, Test batches: {len(test_files)}")
    else:
        print(f"Creating pre-encoded features for fold {fold + 1}...")
        prot_model = EsmForMaskedLM.from_pretrained(args.prot_encoder_path)
        drug_model = AutoModel.from_pretrained(args.drug_encoder_path)
        
        model = TokenEncoder(prot_model, drug_model)
        model = model.to(args.device)
        
        def collate_fn(batch):
            return collate_fn_batch_encoding(batch, prot_tokenizer, drug_tokenizer)
        
        train_examples = dataset_processor.get_train_examples()
        valid_examples = dataset_processor.get_val_examples()
        test_examples = dataset_processor.get_test_examples()

        train_dataloader = DataLoader(
            train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        valid_dataloader = DataLoader(
            valid_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_dataloader = DataLoader(
            test_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        
        train_files = get_feature_fold(model, train_dataloader, args, "train", fold + 1)
        valid_files = get_feature_fold(model, valid_dataloader, args, "valid", fold + 1)
        test_files = get_feature_fold(model, test_dataloader, args, "test", fold + 1)
        
        # Update feature status in dataset processor config
        dataset_processor.update_feature_status(fold, saved=True)
        
        # Clean up encoding models
        del model, prot_model, drug_model
        torch.cuda.empty_cache()
        gc.collect()
    
    return train_files, valid_files, test_files

def get_feature_fold(model, dataloader, args, set_type, fold_num):
    """Pre-encode features and save them to batch files for a specific fold"""
    dataset_name = get_dataset_name(args)
    subdirectory = os.path.join(args.token_cache, dataset_name, f"fold_{fold_num}")
    os.makedirs(subdirectory, exist_ok=True)
    
    batch_files = []
    batch_number = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader), desc=f"Encoding {set_type} fold {fold_num}"):
            if len(batch) == 7:
                prot_ids, prot_mask, drug_ids, drug_mask, labels, prot_seqs, drug_smiles = batch
            else:
                prot_ids, prot_mask, drug_ids, drug_mask, labels = batch
                prot_seqs = None
                drug_smiles = None
                
            prot_ids = prot_ids.to(args.device)
            prot_mask = prot_mask.to(args.device)
            drug_ids = drug_ids.to(args.device)
            drug_mask = drug_mask.to(args.device)
            
            prot_embed, drug_embed = model.encoding(prot_ids, prot_mask, drug_ids, drug_mask)
            prot_embed = prot_embed.cpu()
            drug_embed = drug_embed.cpu()
            prot_mask = prot_mask.cpu()
            drug_mask = drug_mask.cpu()
            labels = labels.cpu()
            
            # Clear GPU tensors
            del prot_ids, drug_ids
            torch.cuda.empty_cache()
        
            # Get dataset name for file naming
            dataset_name = get_dataset_name(args)
            batch_file = os.path.join(
                subdirectory,
                f"{dataset_name}_{set_type}_batch_{batch_number}.pt"
            )
            # Save embeddings and sequences
            save_dict = {
                'prot': prot_embed,
                'drug': drug_embed,
                'prot_mask': prot_mask,
                'drug_mask': drug_mask,
                'y': labels
            }
            
            # Add sequences if available
            if prot_seqs is not None:
                save_dict['prot_seqs'] = prot_seqs
            if drug_smiles is not None:
                save_dict['drug_smiles'] = drug_smiles
                
            torch.save(save_dict, batch_file)
            batch_files.append(batch_file)
            batch_number += 1
    return batch_files

def test(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if len(batch) == 7:
                # Batch includes sequences
                prot, drug, prot_mask, drug_mask, labels, prot_seqs, drug_smiles = batch
                prot, drug = prot.to(device), drug.to(device)
                prot_mask, drug_mask = prot_mask.to(device), drug_mask.to(device)
                labels = labels.to(device)
                output, _ = model(prot, drug, prot_seqs, drug_smiles, labels)
            elif len(batch) == 5:
                # Legacy format without sequences
                prot, drug, prot_mask, drug_mask, labels = batch
                prot, drug = prot.to(device), drug.to(device)
                prot_mask, drug_mask = prot_mask.to(device), drug_mask.to(device)
                labels = labels.to(device)
                output, _ = model(prot, drug, prot_mask, drug_mask, labels)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements. Expected 5 or 7 elements.")
                
            probs = output
            predictions.extend(probs.squeeze().cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            
            # Clean up test batch
            del output
            if 'prot' in locals():
                del prot, drug, prot_mask, drug_mask
            if 'prot_ids' in locals():
                del prot_ids, drug_ids
            del labels
            
    torch.cuda.empty_cache()
    
    auc = roc_auc_score(actuals, predictions)
    aupr = average_precision_score(actuals, predictions)
    accuracy = accuracy_score(actuals, np.array(predictions) > 0.5)
    print(f'Test AUC: {auc:.4f}, AUPR: {aupr:.4f}, Accuracy: {accuracy:.4f}')
    swanlab.log({"Test AUC": auc, "AUPR": aupr, "Accuracy": accuracy})


def main():
    args = parse_config()
    device = torch.device(args.device)
    print(f"Current device: {args.device}.")
    
    swanlab.init(project="DTI_Prediction", config=args, save_code=True)
    swanlab.config.update(args)
    
    # Create save directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Get dataset name from path if it's a file path
    dataset_name = get_dataset_name(args)
    
    best_model_dir = f"{args.save_path}{dataset_name}_kfold_{args.k_folds}_{timestamp_str}/"
    os.makedirs(best_model_dir, exist_ok=True)
    args.save_name = best_model_dir
    
    # Initialize tokenizers
    prot_tokenizer = EsmTokenizer.from_pretrained(args.prot_encoder_path)
    print("prot_tokenizer", len(prot_tokenizer))
    
    drug_tokenizer = AutoTokenizer.from_pretrained(args.drug_encoder_path)
    print("drug_tokenizer", len(drug_tokenizer))
    
    # K-fold cross validation
    print("Using k-fold cross validation")
    dataset_processor = load_dataset(args)
    
    _ = train_kfold(DTIModel, dataset_processor, prot_tokenizer, drug_tokenizer, args, device)
    swanlab.finish()

if __name__ == "__main__":
    main()