import numpy as np
import geopandas as gpd
from sklearn.model_selection import KFold
from typing import Generator, Tuple

def spatial_block_cross_validation(gdf: gpd.GeoDataFrame, n_splits: int = 5, grid_size: float = 0.05) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generates train/validation fold indices grouped into spatial grid blocks to prevent spatial autocorrelation leakage."""
    bounds = gdf.total_bounds
    x_min, y_min, x_max, y_max = bounds
    
    # Assign spatial block identifier to each geometry centroid
    centroids = gdf.geometry.centroid
    col_idx = np.floor((centroids.x - x_min) / grid_size).astype(int)
    row_idx = np.floor((centroids.y - y_min) / grid_size).astype(int)
    block_ids = col_idx * 10000 + row_idx
    
    unique_blocks = np.unique(block_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_blk_idx, val_blk_idx in kf.split(unique_blocks):
        train_blocks = unique_blocks[train_blk_idx]
        val_blocks = unique_blocks[val_blk_idx]
        
        train_mask = np.isin(block_ids, train_blocks)
        val_mask = np.isin(block_ids, val_blocks)
        
        yield np.where(train_mask)[0], np.where(val_mask)[0]
