import argparse
from isca.train import main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the ISCA model")
    parser.add_argument("--config", type=str, default="config/default.yaml", 
                       help="Path to the configuration file")
    args = parser.parse_args()
    
    main(args) 