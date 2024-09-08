import { HardhatRuntimeEnvironment } from "hardhat/types";
import { DeployFunction } from "hardhat-deploy/types";
import { ethers } from "hardhat";

const deployContracts: DeployFunction = async function (hre: HardhatRuntimeEnvironment) {
  const { deployer } = await hre.getNamedAccounts();
  const { deploy } = hre.deployments;

  // Get the signer for the deployer
  const [signer] = await ethers.getSigners();

  // The account to which we'll transfer ownership
  const newOwner = "0xfC5fA9EE7EEA94a038d8f6Ece9DEb419D346BBe4";

  // Deploy ChainlinkPriceOracle
  const chainlinkPriceOracle = await deploy("ChainlinkPriceOracle", {
    from: deployer,
    args: [],
    log: true,
    autoMine: true,
  });

  // Deploy MLPredictionOracle
  const mlPredictionOracle = await deploy("MLPredictionOracle", {
    from: deployer,
    args: [],
    log: true,
    autoMine: true,
  });

  // Deploy UniswapAdapter (using Sepolia Uniswap V3 SwapRouter02)
  const uniswapAdapter = await deploy("UniswapAdapter", {
    from: deployer,
    args: ["0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD"],
    log: true,
    autoMine: true,
  });

  // Deploy SmartPortfolioManager
  const smartPortfolioManager = await deploy("SmartPortfolioManager", {
    from: deployer,
    args: ["Smart Portfolio", "SPT"],
    log: true,
    autoMine: true,
  });

  // Deploy InvestmentVehicle
  const investmentVehicle = await deploy("InvestmentVehicle", {
    from: deployer,
    args: [
      "ETH-WBTC Vehicle",
      "ETHBTC",
      smartPortfolioManager.address,
      uniswapAdapter.address,
      chainlinkPriceOracle.address,
      mlPredictionOracle.address,
    ],
    log: true,
    autoMine: true,
  });

  // Post-deployment configuration
  const chainlinkPriceOracleContract = await ethers.getContractAt(
    "ChainlinkPriceOracle",
    chainlinkPriceOracle.address,
    signer,
  );
  const mlPredictionOracleContract = await ethers.getContractAt(
    "MLPredictionOracle",
    mlPredictionOracle.address,
    signer,
  );
  const smartPortfolioManagerContract = await ethers.getContractAt(
    "SmartPortfolioManager",
    smartPortfolioManager.address,
    signer,
  );
  const investmentVehicleContract = await ethers.getContractAt("InvestmentVehicle", investmentVehicle.address, signer);
  const uniswapAdapterContract = await ethers.getContractAt("UniswapAdapter", uniswapAdapter.address, signer);

  // Configure ChainlinkPriceOracle with Sepolia addresses
  await chainlinkPriceOracleContract.updatePriceFeed(
    "0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9", // Sepolia WBTC
    "0x1b44F3514812d835EB1BDB0acB33d3fA3351Ee43", // Sepolia BTC/USD price feed
  );
  await chainlinkPriceOracleContract.updatePriceFeed(
    "0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9", // Sepolia WETH
    "0x694AA1769357215DE4FAC081bf1f309aDC325306", // Sepolia ETH/USD price feed
  );

  // Grant UPDATER_ROLE to SmartPortfolioManager in MLPredictionOracle
  const UPDATER_ROLE = await mlPredictionOracleContract.UPDATER_ROLE();
  await mlPredictionOracleContract.grantRole(UPDATER_ROLE, smartPortfolioManager.address);

  // Add InvestmentVehicle to SmartPortfolioManager
  await smartPortfolioManagerContract.addVehicle(investmentVehicle.address, "ETH-WBTC Vehicle");

  // Transfer ownership of contracts to the specified account
  await smartPortfolioManagerContract.transferOwnership(newOwner);
  await investmentVehicleContract.transferOwnership(newOwner);
  await uniswapAdapterContract.transferOwnership(newOwner);
  await chainlinkPriceOracleContract.transferOwnership(newOwner);

  // For MLPredictionOracle, we grant the UPDATER_ROLE to the new owner
  await mlPredictionOracleContract.grantRole(UPDATER_ROLE, newOwner);

  console.log("Contracts deployed and configured:");
  console.log("ChainlinkPriceOracle:", chainlinkPriceOracle.address);
  console.log("MLPredictionOracle:", mlPredictionOracle.address);
  console.log("UniswapAdapter:", uniswapAdapter.address);
  console.log("SmartPortfolioManager:", smartPortfolioManager.address);
  console.log("InvestmentVehicle:", investmentVehicle.address);
  console.log("Ownership transferred to:", newOwner);
};

export default deployContracts;

deployContracts.tags = ["SmartPortfolio"];
