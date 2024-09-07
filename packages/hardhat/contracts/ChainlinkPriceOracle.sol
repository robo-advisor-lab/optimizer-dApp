// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@chainlink/contracts/src/v0.8/shared/interfaces/AggregatorV3Interface.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ChainlinkPriceOracle is Ownable {
    mapping(address => address) public priceFeedsMapping;

    event PriceFeedUpdated(address indexed token, address indexed priceFeed);

    constructor() {}

    function updatePriceFeed(address token, address priceFeed) external onlyOwner {
        priceFeedsMapping[token] = priceFeed;
        emit PriceFeedUpdated(token, priceFeed);
    }

    function getLatestPrice(address token) public view returns (int256, uint8) {
        address priceFeedAddress = priceFeedsMapping[token];
        require(priceFeedAddress != address(0), "Price feed not found for token");

        AggregatorV3Interface priceFeed = AggregatorV3Interface(priceFeedAddress);
        (
            uint80 roundID, 
            int256 price,
            uint256 startedAt,
            uint256 timeStamp,
            uint80 answeredInRound
        ) = priceFeed.latestRoundData();
        
        require(timeStamp > 0, "Round not complete");
        
        uint8 decimals = priceFeed.decimals();
        return (price, decimals);
    }

    function getLatestPriceInUSD(address token) external view returns (uint256) {
        (int256 price, uint8 decimals) = getLatestPrice(token);
        require(price > 0, "Invalid price");
        
        // Convert price to 18 decimals (standard for most ERC20 tokens)
        return uint256(price) * 10**(18 - decimals);
    }
}
