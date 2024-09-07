from web3 import Web3
from flask import Flask, jsonify
import brandyns_ml_model  

app = Flask(__name__)

# Set up Web3 connection
w3 = Web3(Web3.HTTPProvider('YOUR_ETHEREUM_NODE_URL'))

# Contract addresses and ABIs
ML_ORACLE_ADDRESS = 'YOUR_ML_PREDICTION_ORACLE_CONTRACT_ADDRESS'
ML_ORACLE_ABI = [...]  # ABI of MLPredictionOracle contract

# Load contract
ml_oracle_contract = w3.eth.contract(address=ML_ORACLE_ADDRESS, abi=ML_ORACLE_ABI)

@app.route('/update_prediction', methods=['POST'])
def update_prediction():
    # Get predictions from your ML model
    assets, weights = your_ml_model.get_predictions()

    # Convert predictions to the format expected by the smart contract
    assets = [Web3.toChecksumAddress(asset) for asset in assets]
    weights = [int(weight * 100) for weight in weights]  # Convert to basis points

    # Call the updatePrediction function of the MLPredictionOracle contract
    tx_hash = ml_oracle_contract.functions.updatePrediction(assets, weights).transact({'from': 'YOUR_ACCOUNT_ADDRESS'})
    
    # Wait for transaction to be mined
    w3.eth.waitForTransactionReceipt(tx_hash)

    return jsonify({"status": "success", "tx_hash": tx_hash.hex()})

if __name__ == '__main__':
    app.run(debug=True)
