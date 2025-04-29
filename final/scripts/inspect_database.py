"""
A simple utility script to inspect SQLite database and show its structure.
This helps diagnose database issues like missing tables.
"""

import os
import sqlite3
import argparse

def inspect_database(db_path):
    """
    Inspect a SQLite database and print information about its structure
    
    Args:
        db_path: Path to the SQLite database file
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' does not exist.")
        return
    
    print(f"\nInspecting database: {db_path}")
    print("-" * 50)
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get database file size
        db_size = os.path.getsize(db_path)
        print(f"Database file size: {db_size/1024:.2f} KB")
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
            return
        
        print(f"\nFound {len(tables)} tables:")
        
        # Examine each table
        for i, (table_name,) in enumerate(tables):
            print(f"\nTable {i+1}: {table_name}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print("  Columns:")
            for col in columns:
                print(f"    {col[1]} ({col[2]})")
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"  Row count: {row_count}")
            
            # Show sample data (first 2 rows)
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                rows = cursor.fetchall()
                
                print("  Sample data (first 2 rows):")
                for row_idx, row in enumerate(rows):
                    print(f"    Row {row_idx+1}:")
                    for col_idx, col_val in enumerate(row):
                        col_name = columns[col_idx][1]
                        # For large text fields, truncate the output
                        if isinstance(col_val, str) and len(col_val) > 100:
                            col_val = col_val[:100] + "... [truncated]"
                        print(f"      {col_name}: {col_val}")
    
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='Inspect a SQLite database')
    parser.add_argument('db_path', nargs='?', default='finance_texts.db',
                      help='Path to the SQLite database file')
    args = parser.parse_args()
    
    inspect_database(args.db_path)

if __name__ == "__main__":
    main()