// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
//import "@openzeppelin/contracts/utils/math/SafeMath.sol";
import "./UniswapAdapter.sol";
import "./ChainlinkPriceOracle.sol";
import "./MLPredictionOracle.sol";

contract InvestmentVehicle is ERC20, ReentrancyGuard, Ownable {
    using SafeMath for uint256;

    struct Asset {
        address tokenAddress;
        uint256 weight; // in basis points (e.g., 5000 = 50%)
        uint256 balance; // current balance of the asset
    }

    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant REBALANCE_INTERVAL = 1 hours;

    address public manager;
    uint256 public nav;
    uint256 public lastRebalance;
    Asset[] public assets;
    UniswapAdapter public uniswapAdapter;
    ChainlinkPriceOracle public priceOracle;
    MLPredictionOracle public mlPredictionOracle;

    event Rebalanced(uint256 timestamp);
    event AssetSwapped(address indexed tokenIn, address indexed tokenOut, uint256 amountIn, uint256 amountOut);

    constructor(
        string memory _name,
        string memory _symbol,
        address _manager,
        address _uniswapAdapter,
        address _priceOracle,
        address _mlPredictionOracle
    ) ERC20(_name, _symbol) {
        manager = _manager;
        lastRebalance = block.timestamp;
        uniswapAdapter = UniswapAdapter(_uniswapAdapter);
        priceOracle = ChainlinkPriceOracle(_priceOracle);
        mlPredictionOracle = MLPredictionOracle(_mlPredictionOracle);
    }

    modifier onlyManagerOrOwner() {
        require(msg.sender == manager || msg.sender == owner(), "Only manager or owner can call this function");
        _;
    }

    // ... [Otras funciones como antes]

    function rebalance() external onlyManagerOrOwner {
        require(block.timestamp >= lastRebalance + REBALANCE_INTERVAL, "Rebalance interval not met");

        uint256 predictionTimestamp = mlPredictionOracle.getPredictionTimestamp();
        require(block.timestamp <= predictionTimestamp + PREDICTION_VALIDITY_PERIOD, "ML prediction is outdated");
        
        uint256 totalValue = getTotalValue();

        for (uint i = 0; i < assets.length; i++) {
            Asset storage asset = assets[i];
            uint256 predictedWeight = mlPredictionOracle.getPrediction(asset.tokenAddress);
            uint256 targetValue = totalValue.mul(predictedWeight).div(BASIS_POINTS);
            uint256 currentValue = getCurrentValue(asset.tokenAddress, asset.balance);
            
            if (currentValue > targetValue) {
                // Sell excess
                uint256 excessAmount = currentValue.sub(targetValue);
                swapAsset(asset.tokenAddress, address(assets[0].tokenAddress), excessAmount);
            } else if (currentValue < targetValue) {
                // Buy more
                uint256 deficitAmount = targetValue.sub(currentValue);
                swapAsset(address(assets[0].tokenAddress), asset.tokenAddress, deficitAmount);
            }

            // Update the asset weight based on the ML prediction
            asset.weight = predictedWeight;
        }

        lastRebalance = block.timestamp;
        emit Rebalanced(lastRebalance);
    }

    function swapAsset(address tokenIn, address tokenOut, uint256 amount) internal {
        IERC20(tokenIn).approve(address(uniswapAdapter), amount);
        uint256 amountOut = uniswapAdapter.swapExactInputSingle(tokenIn, tokenOut, amount, 0);

        // Update balances
        for (uint i = 0; i < assets.length; i++) {
            if (assets[i].tokenAddress == tokenIn) {
                assets[i].balance = assets[i].balance.sub(amount);
            } else if (assets[i].tokenAddress == tokenOut) {
                assets[i].balance = assets[i].balance.add(amountOut);
            }
        }

        emit AssetSwapped(tokenIn, tokenOut, amount, amountOut);
    }

    function getTotalValue() public view returns (uint256) {
        uint256 total = 0;
        for (uint i = 0; i < assets.length; i++) {
            total = total.add(getCurrentValue(assets[i].tokenAddress, assets[i].balance));
        }
        return total;
    }

    function getCurrentValue(address token, uint256 balance) public view returns (uint256) {
        uint256 priceInUSD = priceOracle.getLatestPriceInUSD(token);
        return balance.mul(priceInUSD).div(1e18); // Assuming balance is in token's smallest unit
    }

    function updateAssetBalance(address tokenAddress, uint256 newBalance) external onlyManagerOrOwner {
        for (uint i = 0; i < assets.length; i++) {
            if (assets[i].tokenAddress == tokenAddress) {
                assets[i].balance = newBalance;
                return;
            }
        }
        revert("Asset not found");
    }

    // Additional helper functions and admin functions would be implemented here
}
