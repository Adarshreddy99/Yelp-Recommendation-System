import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import Sequence
import faiss
from collections import defaultdict
import scipy.sparse as sp

# ================= CONFIG =================
CONFIG = {
    "csv_path": "D:/Projects/Yelp-Recommendation-System/data/raw/yelp_filtered_reviews.csv",
    "mlflow_tracking_uri": "https://dagshub.com/Adarshreddy99/Yelp-Recommendation-System.mlflow",
    "dagshub_repo_owner": "Adarshreddy99",
    "dagshub_repo_name": "Yelp-Recommendation-System",
    "experiment_name": "Hybrid_Recommendation_System_TensorFlow",
    "embedding_dim": 64,
    "hidden_dim": 128,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "epochs": 20,
    "test_size": 0.2,
    "min_category_samples": 1000,   # Minimum samples per category to include
    "device": "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
}

# Setup GPU if available
if tf.config.list_physical_devices('GPU'):
    print("GPU available, using GPU")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Using CPU")

# ============= Setup MLflow =============
# (MLflow setup code omitted for brevity)

# ============= Data Loading & Preprocessing =============
def load_data(path):
    """Load and perform initial data validation"""
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic validation
    required_cols = ['user_id', 'business_id', 'stars', 'primary_category']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove null values in essential columns
    df = df.dropna(subset=required_cols)
    print(f"After removing nulls: {df.shape}")
    
    return df

def filter_categories_by_sample_size(df, min_samples=1000):
    """Filter categories to include only those with sufficient samples"""
    print("\nOriginal category distribution:")
    category_counts = df['primary_category'].value_counts()
    print(category_counts)
    
    # Filter categories with sufficient samples
    valid_categories = category_counts[category_counts >= min_samples].index.tolist()
    df_filtered = df[df['primary_category'].isin(valid_categories)].copy()
    
    print(f"\nFiltered to categories with >= {min_samples} samples:")
    print(f"Selected categories: {valid_categories}")
    print(f"Data shape after filtering: {df_filtered.shape}")
    print("\nFiltered category distribution:")
    print(df_filtered['primary_category'].value_counts())
    
    return df_filtered

# --- MODIFIED create_mappings for Global Item Index ---
def create_mappings(df):
    """Create global user and item mappings."""
    # Global user mapping
    user2idx = {u: i for i, u in enumerate(df['user_id'].unique())}
    df['user_idx'] = df['user_id'].map(user2idx)

    # Global item mapping for ALL unique business_ids
    business_ids = df['business_id'].unique()
    item2idx = {item: i for i, item in enumerate(business_ids)}
    df['item_idx'] = df['business_id'].map(item2idx)
    
    # Also keep a reverse mapping from global item_idx to business_id
    idx2item = {i: item for item, i in item2idx.items()}

    # Category-specific item lists (for later lookup in FAISS, not for direct mapping)
    category_item_lists = {}
    for category in df['primary_category'].unique():
        cat_business_ids = df[df['primary_category'] == category]['business_id'].unique()
        category_item_lists[category] = [item2idx[bid] for bid in cat_business_ids]
    
    # Remove any rows where mapping failed (shouldn't happen with global mapping if data is clean)
    df = df.dropna(subset=['user_idx', 'item_idx']).copy()
    
    print(f"Final data shape: {df.shape}")
    print(f"Number of users: {len(user2idx)}")
    print(f"Number of total unique items (globally): {len(item2idx)}")
    for cat, item_idxs in category_item_lists.items():
        print(f"Number of {cat} items (unique in category): {len(item_idxs)}")
    
    return df, user2idx, item2idx, idx2item, category_item_lists

def split_data(df, test_size=0.2):
    """Split data into train and test sets, stratified by category"""
    # Check if we have enough samples in each category for stratification
    category_counts = df['primary_category'].value_counts()
    min_count = category_counts.min()
    
    if min_count < 2:
        print(f"Warning: Some categories have very few samples (min: {min_count})")
        print("Performing random split without stratification")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    else:
        print("Performing stratified split by category")
        train_df, test_df = train_test_split(
            df, test_size=test_size, 
            stratify=df['primary_category'], 
            random_state=42
        )
    
    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Check category distribution in splits
    print("\nTrain set category distribution:")
    print(train_df['primary_category'].value_counts())
    print("\nTest set category distribution:")
    print(test_df['primary_category'].value_counts())
    
    return train_df, test_df

# ============= EDA Functions =============
def perform_eda(df):
    """Comprehensive EDA with MLflow logging"""
    
    # Rating distribution
    plt.figure(figsize=(12, 8))
    
    # Overall rating distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['stars'], bins=5, kde=True)
    plt.title("Overall Rating Distribution")
    plt.xlabel("Stars")
    plt.ylabel("Frequency")
    
    # Rating distribution by category
    plt.subplot(2, 2, 2)
    for category in df['primary_category'].unique():
        cat_data = df[df['primary_category'] == category]
        plt.hist(cat_data['stars'], alpha=0.6, label=category, bins=5)
    plt.title("Rating Distribution by Category")
    plt.xlabel("Stars")
    plt.ylabel("Frequency")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Category distribution
    plt.subplot(2, 2, 3)
    category_counts = df['primary_category'].value_counts()
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    plt.title("Category Distribution")
    
    # Average rating by category
    plt.subplot(2, 2, 4)
    avg_ratings = df.groupby('primary_category')['stars'].mean().sort_values(ascending=False)
    avg_ratings.plot(kind='bar')
    plt.title("Average Rating by Category")
    plt.ylabel("Average Stars")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("comprehensive_eda.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # User and Item activity distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # User activity distribution
    user_counts = df['user_idx'].value_counts()
    sns.histplot(user_counts, bins=50, ax=axes[0,0])
    axes[0,0].set_title("User Activity Distribution")
    axes[0,0].set_xlabel("Number of Reviews per User")
    
    # Log-scale user activity
    sns.histplot(user_counts, bins=50, ax=axes[0,1])
    axes[0,1].set_yscale('log')
    axes[0,1].set_title("User Activity Distribution (Log Scale)")
    
    # Item activity by category (now using global item_idx, but filtered by category)
    categories = df['primary_category'].unique()[:2]  # Show top 2 categories
    for i, category in enumerate(categories):
        cat_data = df[df['primary_category'] == category]
        item_counts = cat_data['item_idx'].value_counts() # These are global item_idxs
        
        ax = axes[1, i]
        sns.histplot(item_counts, bins=30, ax=ax, alpha=0.7)
        ax.set_title(f"{category} Item Activity (Global Index)")
        ax.set_xlabel("Number of Reviews per Item (Global Index)")
    
    plt.tight_layout()
    plt.savefig("activity_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and log sparsity metrics
    sparsity_metrics = {}
    for category in df['primary_category'].unique():
        cat_data = df[df['primary_category'] == category]
        n_users = cat_data['user_idx'].nunique()
        n_items = cat_data['item_idx'].nunique() # These are count of global items unique to this category
        n_ratings = len(cat_data)
        
        # Sparsity calculation: total possible interactions within this category's users and items
        # This is tricky with a global item_idx if you want 'category-specific' sparsity precisely.
        # A more accurate sparsity for a specific category would be based on (unique users in category * unique items in category)
        sparsity = 1 - (n_ratings / (n_users * n_items))
        sparsity_metrics[f'{category}_sparsity'] = sparsity
        
        print(f"{category}:")
        print(f"  Users: {n_users}, Items: {n_items}, Ratings: {n_ratings}")
        print(f"  Sparsity: {sparsity:.4f}")
    
    # Overall statistics
    overall_stats = {
        'avg_rating': df['stars'].mean(),
        'rating_std': df['stars'].std(),
        'num_categories': df['primary_category'].nunique(),
        'total_users': df['user_idx'].nunique(),
        'total_items': df['business_id'].nunique(), # This is the true global item count
        'total_ratings': len(df)
    }
    
    print("\nOverall Statistics:")
    for key, value in overall_stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    return {**sparsity_metrics, **overall_stats}

# ============= Model Architecture =============
def create_user_tower(num_users, emb_dim=64, hidden_dim=128):
    """Create the global user embedding tower."""
    user_input = layers.Input(shape=(), name='global_user_input')
    
    user_embedding = layers.Embedding(
        input_dim=num_users,
        output_dim=emb_dim,
        name='global_user_embedding'
    )(user_input)
    
    user_vec = layers.Flatten()(user_embedding)
    
    x = layers.Dense(hidden_dim, activation='relu', name='global_user_dense_1')(user_vec)
    x = layers.BatchNormalization(name='global_user_bn_1')(x)
    x = layers.Dropout(0.2, name='global_user_dropout_1')(x)
    user_output = layers.Dense(emb_dim, activation='linear', name='global_user_output')(x)
    
    return Model(inputs=user_input, outputs=user_output, name='user_tower')

def create_item_tower(num_total_items, emb_dim=64, hidden_dim=128):
    """Create the global item embedding tower."""
    item_input = layers.Input(shape=(), name='global_item_input')
    
    item_embedding = layers.Embedding(
        input_dim=num_total_items, # Now uses the total unique items globally
        output_dim=emb_dim,
        name='global_item_embedding'
    )(item_input)
    
    item_vec = layers.Flatten()(item_embedding)
    
    x = layers.Dense(hidden_dim, activation='relu', name='global_item_dense_1')(item_vec)
    x = layers.BatchNormalization(name='global_item_bn_1')(x)
    x = layers.Dropout(0.2, name='global_item_dropout_1')(x)
    item_output = layers.Dense(emb_dim, activation='linear', name='global_item_output')(x)
    
    return Model(inputs=item_input, outputs=item_output, name='item_tower')

def create_hybrid_two_tower_models(num_users, num_total_items, categories, emb_dim=64, hidden_dim=128):
    """
    Create category-specific models using shared GLOBAL user and item towers.
    Each category-specific model will have its own prediction head.
    """
    
    # Create global user and item towers (these will be shared)
    user_tower = create_user_tower(num_users, emb_dim, hidden_dim)
    item_tower = create_item_tower(num_total_items, emb_dim, hidden_dim)
    
    category_prediction_models = {}
    
    # Define inputs for the overall hybrid model (user_idx, item_idx, category)
    global_user_input = layers.Input(shape=(), name='user_idx_input')
    global_item_input = layers.Input(shape=(), name='item_idx_input')
    
    # Get global user and item embeddings from their respective towers
    user_embedding = user_tower(global_user_input) # Output from user tower
    item_embedding = item_tower(global_item_input) # Output from item tower

    # Common layers after tower embeddings (optional, but good for shared representation learning)
    merged_embeddings = layers.Concatenate(name='concatenated_embeddings')([user_embedding, item_embedding])
    
    # For category-specific prediction heads, we need a way to route or specialize.
    # The most straightforward way with separate models is to compile each model
    # with the relevant data, but the towers are shared.
    # To enable "category-specific models" as you desire, we will define a
    # "core model" that takes global embeddings and outputs a final rating,
    # and then create separate instances of this core model for each category,
    # allowing their final dense layers to be trained independently.

    # This requires a slightly different approach for training:
    # Instead of truly separate models, we define ONE model with shared towers
    # and then train it with category as a feature, OR
    # We train category-specific models where the *towers* are frozen or
    # fine-tuned and the final layers are category-specific.

    # Given your current training loop, creating separate models that each
    # encapsulate the global towers and their own prediction heads is the most
    # direct translation. The crucial part is ensuring the *towers* are the same Keras model instances.

    for category in categories:
        print(f"Creating hybrid model for {category}...")
        
        # Inputs to this specific category's prediction model
        # These are just placeholder inputs for the model definition;
        # the actual tower outputs will be fed.
        user_input_cat = layers.Input(shape=(), name=f'{category}_user_input_placeholder')
        item_input_cat = layers.Input(shape=(), name=f'{category}_item_input_placeholder')

        # Pass inputs through the shared towers
        # We'll use the functional API here to tie these inputs to the *same* tower instances
        user_emb_output = user_tower(user_input_cat)
        item_emb_output = item_tower(item_input_cat)

        # Concatenate embeddings
        combined = layers.Concatenate(name=f'{category}_combined_embeddings')([user_emb_output, item_emb_output])
        
        # Category-specific prediction head
        prediction_head = layers.Dense(hidden_dim, activation='relu', name=f'{category}_pred_dense_1')(combined)
        prediction_head = layers.BatchNormalization(name=f'{category}_pred_bn_1')(prediction_head)
        prediction_head = layers.Dropout(0.2, name=f'{category}_pred_dropout_1')(prediction_head)
        prediction_head = layers.Dense(hidden_dim // 2, activation='relu', name=f'{category}_pred_dense_2')(prediction_head)
        prediction_head = layers.Dropout(0.1, name=f'{category}_pred_dropout_2')(prediction_head)
        raw_rating_output = layers.Dense(1, activation='sigmoid', name=f'{category}_raw_output')(prediction_head)
        scaled_rating_output = layers.Lambda(lambda x: x * 4 + 1, name=f'{category}_rating_scale')(raw_rating_output)
        
        # Create the category-specific model
        model = Model(inputs=[user_input_cat, item_input_cat], outputs=scaled_rating_output, name=f'{category}_hybrid_prediction_model')
        category_prediction_models[category] = model
        
        print(f"Hybrid prediction model for {category} created with {model.count_params():,} parameters")
    
    return user_tower, item_tower, category_prediction_models

# ============= Training Functions =============
def train_models(models_info, train_df, test_df, epochs=20, batch_size=1024):
    """Train category-specific prediction models, sharing global towers."""
    
    user_tower, item_tower, category_prediction_models = models_info
    trained_models = {}
    
    # Compile the towers here once, if you want them to be trained as part of the overall process
    # Or, if you want to freeze them, compile them with a null optimizer or set trainable=False.
    # For a 'hybrid' model, they should train with the prediction heads.
    # We will compile each full category_prediction_model, which implicitly trains the shared towers.

    for category, model in category_prediction_models.items():
        print(f"\n{'='*50}")
        print(f"Training {category} model (using shared towers)...")
        print(f"{'='*50}")
        
        # Filter data for this category
        train_cat = train_df[train_df['primary_category'] == category].copy()
        test_cat = test_df[test_df['primary_category'] == category].copy()
        
        if len(train_cat) == 0:
            print(f"No training data for {category}, skipping...")
            continue
            
        if len(test_cat) == 0:
            print(f"No test data for {category}, using 10% of training data for validation...")
            train_cat, test_cat = train_test_split(train_cat, test_size=0.1, random_state=42)
        
        print(f"Training samples: {len(train_cat)}")
        print(f"Test samples: {len(test_cat)}")
        
        # Prepare data (using global user_idx and item_idx)
        X_train = [train_cat['user_idx'].values, train_cat['item_idx'].values]
        y_train = train_cat['stars'].values
        
        X_test = [test_cat['user_idx'].values, test_cat['item_idx'].values]
        y_test = test_cat['stars'].values
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        trained_models[category] = {
            'model': model,
            'history': history
        }
        
        final_loss = history.history['val_loss'][-1]
        final_mae = history.history['val_mae'][-1]
        
        print(f"{category} - Final Val Loss: {final_loss:.4f}, Final Val MAE: {final_mae:.4f}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{category} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'{category} - MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{category}_training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    return trained_models

def evaluate_models(trained_models, test_df):
    """Evaluate all models"""
    overall_predictions = []
    overall_actuals = []
    category_metrics = {}
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    
    for category, model_info in trained_models.items():
        model = model_info['model']
        
        # Filter test data for this category
        test_cat = test_df[test_df['primary_category'] == category]
        
        if len(test_cat) == 0:
            print(f"No test data for {category}, skipping evaluation...")
            continue
        
        # Prepare test data (using global user_idx and item_idx)
        X_test = [test_cat['user_idx'].values, test_cat['item_idx'].values]
        y_test = test_cat['stars'].values
        
        # Make predictions
        predictions = model.predict(X_test, verbose=0)
        predictions = predictions.flatten()
        
        # Clip predictions to valid range
        predictions = np.clip(predictions, 1, 5)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        category_metrics[category] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'samples': len(test_cat)
        }
        
        overall_predictions.extend(predictions)
        overall_actuals.extend(y_test)
        
        print(f"{category:20} - Samples: {len(test_cat):6d} | MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        
    # Overall metrics
    if overall_predictions:
        overall_mse = mean_squared_error(overall_actuals, overall_predictions)
        overall_mae = mean_absolute_error(overall_actuals, overall_predictions)
        overall_rmse = np.sqrt(overall_mse)
        
        print(f"{'='*50}")
        print(f"{'OVERALL':20} - Samples: {len(overall_predictions):6d} | MSE: {overall_mse:.4f} | MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f}")
        print(f"{'='*50}")
        
        return {
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'category_metrics': category_metrics
        }
    
    return {'category_metrics': category_metrics}

# ============= FAISS Index Functions =============
# ... (previous code) ...

# ============= FAISS Index Functions =============
def build_faiss_indices(user_tower, item_tower, category_item_lists):
    """
    Build FAISS indices for each category.
    Item embeddings are extracted from the GLOBAL item tower.
    """
    indices = {}
    
    # Get all item embeddings from the global item tower once
    # --- FIX IS HERE ---
    # Access the input_dim from the Embedding layer within the item_tower
    # item_tower.layers[0] is the Input layer
    # item_tower.layers[1] should be the Embedding layer (check your model definition if this index changes)
    try:
        embedding_layer = item_tower.get_layer(name='global_item_embedding')
        num_total_items_for_embedding = embedding_layer.input_dim
    except ValueError:
        # Fallback if layer name is not found, assume it's the second layer (index 1)
        print("Warning: 'global_item_embedding' layer not found by name, falling back to index 1.")
        num_total_items_for_embedding = item_tower.layers[1].input_dim
        
    # Generate all item IDs from 0 up to num_total_items_for_embedding - 1
    all_item_ids = np.arange(num_total_items_for_embedding)
    
    print(f"Extracting all {len(all_item_ids)} item embeddings from global item tower...")
    # Ensure the item_tower is built before calling predict
    # This might happen if it hasn't been called with data during training yet
    if not item_tower.built:
        # A small dummy call to build the model if it hasn't been already
        _ = item_tower(np.array([0])) 
        print("Item tower was not built, forced building with a dummy input.")

    all_item_embeddings = item_tower.predict(all_item_ids, verbose=0) # Added verbose=0 to suppress output
    print(f"Shape of all item embeddings: {all_item_embeddings.shape}")

    for category, item_global_idxs in category_item_lists.items():
        print(f"Building FAISS index for {category} with {len(item_global_idxs)} items...")
        
        # Select only the embeddings relevant to this category
        # Ensure item_global_idxs is a list of valid integer indices
        if not item_global_idxs:
            print(f"No items found for category {category}, skipping FAISS index creation.")
            continue
            
        # Convert to numpy array for advanced indexing
        item_global_idxs_array = np.array(item_global_idxs, dtype=int)
        
        # Check for any out-of-bounds indices if needed (optional)
        if np.max(item_global_idxs_array) >= num_total_items_for_embedding or np.min(item_global_idxs_array) < 0:
            print(f"Warning: Out-of-bounds item_global_idxs detected for {category}. Skipping.")
            continue

        category_item_embeddings = all_item_embeddings[item_global_idxs_array]
        
        # Build FAISS index
        # Ensure the embedding dimension is positive
        if category_item_embeddings.shape[0] == 0:
            print(f"No embeddings to add for category {category}, skipping FAISS index creation.")
            continue

        if category_item_embeddings.shape[1] <= 0:
            raise ValueError(f"Embedding dimension for {category} is not positive: {category_item_embeddings.shape[1]}")

        index = faiss.IndexFlatL2(category_item_embeddings.shape[1])
        index.add(category_item_embeddings.astype('float32'))
        indices[category] = {
            'index': index,
            'embedding_dim': category_item_embeddings.shape[1],
            'item_global_idxs': item_global_idxs # Store original global indices (as list or array)
        }
        
        print(f"Built FAISS index for {category}: {category_item_embeddings.shape}")
    
    return indices


def get_recommendations(user_id, category, faiss_index_info, idx2item, user_tower, k=10):
    """Get top-k recommendations for a user in a specific category using global towers."""
    try:
        # Get user embedding from the global user tower
        user_embedding = user_tower.predict(np.array([user_id]))
        user_embedding = user_embedding.flatten()
        
        # Search in FAISS index specific to the category
        user_embedding = user_embedding.astype('float32').reshape(1, -1)
        distances, item_local_indices = faiss_index_info['index'].search(user_embedding, k)
        
        # Map back to original global item_idxs then to business IDs
        # The 'item_local_indices' from FAISS refer to the index within the *category_item_embeddings* array
        # We need to map these back to the global item_idx using `item_global_idxs` stored in faiss_index_info
        
        recommended_global_item_idxs = [faiss_index_info['item_global_idxs'][idx] for idx in item_local_indices[0]]
        recommended_businesses = [idx2item.get(idx, f"Unknown_{idx}") for idx in recommended_global_item_idxs]
        
        return distances[0], recommended_global_item_idxs, recommended_businesses
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return [], [], []

# ============= Main Pipeline =============
def main_pipeline():
    """Execute the complete pipeline"""
    
    print("Starting Hybrid Two-Tower Recommendation System Pipeline (TensorFlow)...")
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = load_data(CONFIG["csv_path"])
    
    # Filter categories by sample size
    df = filter_categories_by_sample_size(df, CONFIG["min_category_samples"])
    
    # Create mappings (user2idx global, item2idx global, category_item_lists for filtering)
    df, user2idx, item2idx, idx2item, category_item_lists = create_mappings(df)
    
    # 2. EDA
    print("\n2. Performing EDA...")
    eda_metrics = perform_eda(df)
    
    # 3. Split data
    print("\n3. Splitting data...")
    train_df, test_df = split_data(df, CONFIG["test_size"])
    
    # 4. Create models (shared towers, category-specific prediction heads)
    print("\n4. Creating models...")
    num_total_items = len(item2idx)
    all_categories = df['primary_category'].unique()

    user_tower, item_tower, category_prediction_models = create_hybrid_two_tower_models(
        num_users=len(user2idx),
        num_total_items=num_total_items,
        categories=all_categories,
        emb_dim=CONFIG["embedding_dim"],
        hidden_dim=CONFIG["hidden_dim"]
    )
    
    # Log model parameters
    total_params = user_tower.count_params() + item_tower.count_params() + \
                   sum(model.count_params() for model in category_prediction_models.values())
    print(f"Models initialized with {total_params:,} total parameters")
    
    # 5. Train models
    print("\n5. Training models...")
    # Pass user_tower, item_tower, and category_prediction_models to training function
    trained_category_models = train_models(
        (user_tower, item_tower, category_prediction_models), train_df, test_df, 
        epochs=CONFIG["epochs"], 
        batch_size=CONFIG["batch_size"]
    )
    
    # 6. Evaluate models
    print("\n6. Evaluating models...")
    evaluation_results = evaluate_models(trained_category_models, test_df)
    
    # 7. Save models
    print("\n7. Saving models...")
    user_tower.save('user_tower.h5')
    item_tower.save('item_tower.h5')
    for category, model_info in trained_category_models.items():
        model_path = f'{category.replace(" ", "_").replace("&", "and")}_prediction_head.h5'
        model_info['model'].save(model_path)
    
    # 8. Build FAISS indices
    print("\n8. Building FAISS indices...")
    # Build FAISS indices using the globally trained item tower and category_item_lists
    faiss_indices = build_faiss_indices(user_tower, item_tower, category_item_lists)
    
    # Save FAISS indices
    for category, index_info in faiss_indices.items():
        safe_category = category.replace(" ", "_").replace("&", "and")
        index_path = f'faiss_index_{safe_category}.index'
        faiss.write_index(index_info['index'], index_path)
        # You might also want to save item_global_idxs for reconstruction
        np.save(f'faiss_index_{safe_category}_global_item_idxs.npy', np.array(index_info['item_global_idxs']))
    
    # 9. Demo recommendations
    print("\n9. Generating sample recommendations...")
    sample_user_id = 0  # Use user_idx 0 (mapped from original user_id)
    
    # Find a user_idx that actually exists in the data for demonstration
    if len(user2idx) > 0:
        sample_user_id = list(user2idx.values())[0] # Get the first mapped user_idx
        print(f"Using sample user_idx: {sample_user_id}")
    else:
        print("No users found for sample recommendations.")
        return trained_category_models, item2idx, faiss_indices, user_tower, item_tower

    for category in trained_category_models.keys():
        if category in faiss_indices:
            distances, item_global_indices, business_ids = get_recommendations(
                sample_user_id, 
                category, 
                faiss_indices[category], 
                idx2item, # Pass global idx2item
                user_tower,
                k=5
            )
            
            print(f"\nTop 5 {category} recommendations for user_idx {sample_user_id}:")
            if business_ids:
                for i, (dist, item_global_idx, business_id) in enumerate(zip(distances, item_global_indices, business_ids)):
                    print(f"  {i+1}. Business: {business_id} (Global Item ID: {item_global_idx}, Distance: {dist:.4f})")
            else:
                print(f"  No recommendations found for {category}.")
        else:
            print(f"\nSkipping recommendations for {category} as FAISS index not built.")
            
    print("\nPipeline completed successfully!")
    
    return trained_category_models, item2idx, faiss_indices, user_tower, item_tower

# ============= Execute Pipeline =============
if __name__ == "__main__":
    trained_models, item2idx, faiss_indices, user_tower, item_tower = main_pipeline()