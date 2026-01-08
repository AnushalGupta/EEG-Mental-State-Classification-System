import mysql.connector
import config

# Database Configuration
DB_CONFIG = config.DB_CONFIG

def get_db_connection():
    """Establishes and returns a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        raise

def create_table_if_not_exists(feature_names):
    """
    Creates the 'eeg_features' table if it doesn't exist.
    Dynamically generates columns based on the feature_names list.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Base columns
    columns = [
        "id INT AUTO_INCREMENT PRIMARY KEY",
        "dataset_name VARCHAR(50)",
        "subject_id VARCHAR(100)",
        "label INT",  # 0: Focused, 1: Unfocused, 2: Drowsy
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ]

    # Add feature columns (FLOAT)
    for feat in feature_names:
        columns.append(f"`{feat}` FLOAT")

    # Construct CREATE TABLE statement
    create_stmt = f"""
    CREATE TABLE IF NOT EXISTS eeg_features (
        {', '.join(columns)}
    );
    """
    
    try:
        cursor.execute(create_stmt)
        conn.commit()
        print("Table 'eeg_features' checked/created successfully.")
    except mysql.connector.Error as err:
        print(f"Error creating table: {err}")
    finally:
        cursor.close()
        conn.close()
