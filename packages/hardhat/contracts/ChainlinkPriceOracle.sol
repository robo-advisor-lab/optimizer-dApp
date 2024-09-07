// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@chainlink/contracts/src/v0.8/shared/interfaces/AggregatorV3Interface.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ChainlinkPriceOracle is Ownable {
    mapping(address => address) public priceFeedsMapping;

    event PriceFeedUpdated(address indexed token, address indexed priceFeed);
    event PriceRequested(address indexed token, uint256 price);

    constructor() Ownable(msg.sender) {
        // Pre-configure some popular tokens (example addresses for Sepolia)
        updatePriceFeed(0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984, 0x694AA1769357215DE4FAC081bf1f309aDC325306); // UNI/USD
        updatePriceFeed(0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619, 0x0715A7794a1dc8e42615F059dD6e406A6594651A); // ETH/USD
        updatePriceFeed(0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599, 0x1b44F3514812d835EB1BDB0acB33d3fA3351Ee43); // BTC/USD
    }

    function updatePriceFeed(address token, address priceFeed) public onlyOwner {
        require(token != address(0), "Invalid token address");
        require(priceFeed != address(0), "Invalid price feed address");
        priceFeedsMapping[token] = priceFeed;
        emit PriceFeedUpdated(token, priceFeed);
    }

    function getLatestPrice(address token) public view returns (int256, uint8) {
        address priceFeedAddress = priceFeedsMapping[token];
        require(priceFeedAddress != address(0), "Price feed not found for token");

        AggregatorV3Interface priceFeed = AggregatorV3Interface(priceFeedAddress);
        (
            /* uint80 roundID */,
            int256 price,
            /* uint256 startedAt */,
            uint256 timeStamp,
            /* uint80 answeredInRound */
        ) = priceFeed.latestRoundData();
        
        require(timeStamp > 0, "Round not complete");
        require(price > 0, "Invalid price");
        
        uint8 decimals = priceFeed.decimals();
        return (price, decimals);
    }

    function getLatestPriceInUSD(address token) public view returns (uint256) {
        (int256 price, uint8 decimals) = getLatestPrice(token);
        uint256 priceInUSD = uint256(price) * 10**(18 - decimals);
        // Remove the emit statement to keep the function as view
        // emit PriceRequested(token, priceInUSD);
        return priceInUSD;
    }

    function getMultiplePricesInUSD(address[] calldata tokens) external view returns (uint256[] memory) {
        uint256[] memory prices = new uint256[](tokens.length);
        for (uint i = 0; i < tokens.length; i++) {
            prices[i] = getLatestPriceInUSD(tokens[i]);
        }
        return prices;
    }

    function isPriceFeedSet(address token) public view returns (bool) {
        return priceFeedsMapping[token] != address(0);
    }
}
