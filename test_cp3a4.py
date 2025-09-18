#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, './unimol')

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC
from unicore import checkpoint_utils, tasks, utils as unicore_utils
from argparse import Namespace
from sklearn.manifold import TSNE


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("test_cp3a4")

def generate_points_in_circle(n, r=1.0, shift=(0,0)):
    theta = np.random.uniform(0, 2*np.pi, n)
    rad = np.sqrt(np.random.uniform(0, 1, n)) * r
    x = rad * np.cos(theta) + shift[0]
    y = rad * np.sin(theta) + shift[1]
    return x, y

def calculate_metrics(y_true, y_score):
    scores = np.expand_dims(y_score, axis=1)
    y_true_expanded = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true_expanded), axis=1)
    scores = scores[scores[:,0].argsort()[::-1]]
    
    bedroc = CalcBEDROC(scores, 1, 80.5)
    auc = CalcAUC(scores, 1)
    
    return auc, bedroc

def main():
    print("=== CP3A4 DrugCLIP Inference ===")
    
    checkpoint_path = "./checkpoint_best.pt"
    target = "cp3a4"
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
    
    saved_args = state['args']
    
    args = Namespace(
        arch='drugclip',
        task='drugclip',
        
        data="./data",
        seed=1,
        max_seq_len=getattr(saved_args, 'max_seq_len', 512),
        max_pocket_atoms=getattr(saved_args, 'max_pocket_atoms', 256),
        
        finetune_mol_model=None,
        finetune_pocket_model=None,
        
        fp16=torch.cuda.is_available(),
        cpu=not torch.cuda.is_available(),
        device_id=0,
        
        mol=getattr(saved_args, 'mol', None),
        pocket=getattr(saved_args, 'pocket', None),
        dist_threshold=getattr(saved_args, 'dist_threshold', 8.0),
        recycling=getattr(saved_args, 'recycling', 1),
        
        loss='in_batch_softmax',
        test_model=False,
        reg=False,
    )
    
    for attr in dir(saved_args):
        if not attr.startswith('_') and not hasattr(args, attr):
            setattr(args, attr, getattr(saved_args, attr))
    
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)
    
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)
    
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()
    
    logger.info("Model loaded successfully")
    model.eval()
    
    print("Running inference for CP3A4 target...")
    with torch.no_grad():
        mol_data_path = f"./data/DUD-E/raw/all/{target}/mols.lmdb"
        mol_dataset = task.load_mols_dataset(mol_data_path, "atoms", "coordinates")
        mol_loader = torch.utils.data.DataLoader(
            mol_dataset, batch_size=64, collate_fn=mol_dataset.collater
        )
        
        mol_reps = []
        labels = []
        
        for sample in mol_loader:
            if use_cuda:
                sample = unicore_utils.move_to_cuda(sample)
            
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            labels.extend(sample["target"].detach().cpu().numpy())
        
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        
        pocket_data_path = f"./data/DUD-E/raw/all/{target}/pocket.lmdb"
        pocket_dataset = task.load_pockets_dataset(pocket_data_path)
        pocket_loader = torch.utils.data.DataLoader(
            pocket_dataset, batch_size=64, collate_fn=pocket_dataset.collater
        )
        
        pocket_reps = []
        for sample in pocket_loader:
            if use_cuda:
                sample = unicore_utils.move_to_cuda(sample)
            
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
        
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        
        similarities = pocket_reps @ mol_reps.T
        scores = similarities.max(axis=0)
        
        auc, bedroc = calculate_metrics(labels, scores)
    
    n_positive = np.sum(labels)
    n_negative = len(labels) - n_positive
    
    print("\n==== CP3A4 Final Results ====")
    print(f"AUROC: {auc:.4f}")
    print(f"BEDROC: {bedroc:.4f}")
    print(f"Positive molecules: {n_positive}")
    print(f"Negative molecules: {n_negative}")
    
    print("\nGenerating visualization...")
    
    _, ax = plt.subplots(figsize=(8, 8))
    
    avg_pocket_rep = pocket_reps.mean(axis=0, keepdims=True)
    
    all_reps = np.vstack([avg_pocket_rep, mol_reps])
    
    print("Applying t-SNE for visualization...")
    tsne = TSNE(n_components=2, 
                perplexity=40,
                learning_rate=200.0,
                random_state=42, 
                max_iter=2000,
                init='pca',
                metric='cosine')
    coords_2d = tsne.fit_transform(all_reps)
    
    pocket_coord = coords_2d[0]
    mol_coords = coords_2d[1:]
    
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    pos_coords = mol_coords[pos_indices]
    neg_coords = mol_coords[neg_indices]
    
    ax.scatter(neg_coords[:, 0], neg_coords[:, 1], c="#fae6cdeb", s=15, alpha=0.8, label="Negative molecule")
    
  
    ax.scatter(pos_coords[:, 0], pos_coords[:, 1], c="#6aaed6", s=20, alpha=0.9, label="Positive molecule")
    
    ax.scatter(pocket_coord[0], pocket_coord[1], marker="*", c="#971313", edgecolors="black", 
              s=100, linewidths=1.0, label="Pocket")
    
    all_x = np.concatenate([neg_coords[:, 0], pos_coords[:, 0], [pocket_coord[0]]])
    all_y = np.concatenate([neg_coords[:, 1], pos_coords[:, 1], [pocket_coord[1]]])
    
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1
    

    ax.set_xlim([all_x.min() - x_margin, all_x.max() + x_margin])
    ax.set_ylim([all_y.min() - y_margin, all_y.max() + y_margin])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    
    output_path = "cp3a4_drugclip.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    plt.close()

if __name__ == "__main__":
    main()