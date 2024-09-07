import { HardhatRuntimeEnvironment } from "hardhat/types";
import { DeployFunction } from "hardhat-deploy/types";

const deployContracts: DeployFunction = async function (hre: HardhatRuntimeEnvironment) {
  const { deployer } = await hre.getNamedAccounts();
  const { deploy } = hre.deployments;

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

  // Deploy UniswapAdapter
  // Note: You'll need to replace args :['UNISWAP_ROUTER_ADDRESS'] with the actual address of the Uniswap router, currently Sepolia
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
      "Investment Vehicle",
      "INV",
      smartPortfolioManager.address,
      uniswapAdapter.address,
      chainlinkPriceOracle.address,
      mlPredictionOracle.address,
    ],
    log: true,
    autoMine: true,
  });

  // You can add any post-deployment configuration here
  // For example, setting up roles, initializing state, etc.

  console.log("Contracts deployed:");
  console.log("ChainlinkPriceOracle:", chainlinkPriceOracle.address);
  console.log("MLPredictionOracle:", mlPredictionOracle.address);
  console.log("UniswapAdapter:", uniswapAdapter.address);
  console.log("SmartPortfolioManager:", smartPortfolioManager.address);
  console.log("InvestmentVehicle:", investmentVehicle.address);
};

export default deployContracts;

deployContracts.tags = ["SmartPortfolio"];
