// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

contract MLPredictionOracle is AccessControl, Pausable {
    bytes32 public constant UPDATER_ROLE = keccak256("UPDATER_ROLE");

    struct Prediction {
        uint256 timestamp;
        mapping(address => uint256) assetWeights; // Asset address to weight (in basis points)
        bytes32 tablelandHash; // Hash of the Tableland row containing detailed prediction data
    }

    Prediction public latestPrediction;
    uint256 public constant BASIS_POINTS = 10000;

    event PredictionUpdated(uint256 timestamp, bytes32 tablelandHash);

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(UPDATER_ROLE, msg.sender);
    }

    function updatePrediction(
        address[] memory assets, 
        uint256[] memory weights, 
        bytes32 tablelandHash
    ) external onlyRole(UPDATER_ROLE) whenNotPaused {
        require(assets.length == weights.length, "Arrays length mismatch");
        require(tablelandHash != bytes32(0), "Invalid Tableland hash");
        
        uint256 totalWeight = 0;
        for (uint i = 0; i < weights.length; i++) {
            totalWeight += weights[i];
        }
        require(totalWeight == BASIS_POINTS, "Total weight must be 10000 basis points");

        latestPrediction.timestamp = block.timestamp;
        latestPrediction.tablelandHash = tablelandHash;
        for (uint i = 0; i < assets.length; i++) {
            latestPrediction.assetWeights[assets[i]] = weights[i];
        }

        emit PredictionUpdated(block.timestamp, tablelandHash);
    }

    function getPrediction(address asset) external view returns (uint256, uint256, bytes32) {
        return (
            latestPrediction.assetWeights[asset],
            latestPrediction.timestamp,
            latestPrediction.tablelandHash
        );
    }

    function getLatestPredictionData() external view returns (uint256, bytes32) {
        return (latestPrediction.timestamp, latestPrediction.tablelandHash);
    }

    function verifyPrediction(
        address[] memory assets, 
        uint256[] memory weights, 
        bytes32 tablelandHash
    ) external view returns (bool) {
        if (tablelandHash != latestPrediction.tablelandHash) {
            return false;
        }

        for (uint i = 0; i < assets.length; i++) {
            if (latestPrediction.assetWeights[assets[i]] != weights[i]) {
                return false;
            }
        }

        return true;
    }

    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
}
