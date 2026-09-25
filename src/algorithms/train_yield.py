import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.algorithms.spatial_cv import spatial_block_cross_validation

def run_training_pipeline(geojson_path: str, feature_columns: list, target_col: str):
    gdf = gpd.read_file(geojson_path)
    X = gdf[feature_columns].values
    y = gdf[target_col].values
    
    rmses, r2s = [], []
    for train_idx, val_idx in spatial_block_cross_validation(gdf, n_splits=5):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        r2s.append(r2_score(y_val, preds))
        
    print(f"Spatial CV Mean RMSE: {np.mean(rmses):.3f} | Mean R2: {np.mean(r2s):.3f}")

if __name__ == "__main__":
    print("Execute with sample parcel features dataset.")
