import os
import glob
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datetime import datetime

def backfill():
    load_dotenv()
    uri = os.environ.get("TIMESCALEDB_URI")
    if not uri:
        print("No TIMESCALEDB_URI found in .env")
        return
        
    # sqlalchemy needs postgresql:// instead of postgres:// 
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql://", 1)
        
    engine = create_engine(uri)
    
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE sensor_history;")
    
    csv_files = glob.glob("data/history/*_history.csv")
    if not csv_files:
        print("No historical CSV files found in data/history/")
        return
        
    print(f"Found {len(csv_files)} station history files. Backfilling to TimescaleDB...")
    
    total_rows = 0
    for f in csv_files:
        station_id = os.path.basename(f).replace("_history.csv", "")
        print(f"Processing {station_id}...")
        
        try:
            df = pd.read_csv(f)
            
            # Map CSV columns to TimescaleDB schema
            # CSV has: timestamp, temperature_c, pressure_hpa, humidity_pct, is_anomaly, fault_type, severity, anomaly_score_pct, suggested_temperature_c, suggested_pressure_hpa, suggested_humidity_pct, health_status, source
            
            df['station_id'] = station_id
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
            
            # Ensure proper boolean types
            if 'is_anomaly' in df.columns:
                df['is_anomaly'] = df['is_anomaly'].astype(bool)
                
            # Drop any accidental duplicates in the CSV that would violate the database UNIQUE constraint
            df = df.drop_duplicates(subset=['station_id', 'timestamp', 'source'])
                
            # Write to timescale
            df.to_sql('sensor_history', engine, if_exists='append', index=False, method='multi', chunksize=1000)
            total_rows += len(df)
            
        except Exception as e:
            print(f"Error processing {station_id}: {e}")
            
    print(f"\nSuccess! Backfilled {total_rows} historical readings into TimescaleDB.")

if __name__ == "__main__":
    backfill()
