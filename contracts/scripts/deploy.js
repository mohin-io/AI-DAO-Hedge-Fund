const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Deploying AI DAO Hedge Fund Smart Contracts...\n");

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with account:", deployer.address);
  console.log("Account balance:", (await deployer.provider.getBalance(deployer.address)).toString());
  console.log();

  // Deploy DAOGovernance
  console.log("📜 Deploying DAOGovernance...");
  const DAOGovernance = await hre.ethers.getContractFactory("DAOGovernance");
  const daoGovernance = await DAOGovernance.deploy();
  await daoGovernance.waitForDeployment();
  const daoAddress = await daoGovernance.getAddress();
  console.log("✅ DAOGovernance deployed to:", daoAddress);
  console.log();

  // Deploy TreasuryManager
  console.log("💰 Deploying TreasuryManager...");
  const TreasuryManager = await hre.ethers.getContractFactory("TreasuryManager");
  const treasuryManager = await TreasuryManager.deploy(daoAddress);
  await treasuryManager.waitForDeployment();
  const treasuryAddress = await treasuryManager.getAddress();
  console.log("✅ TreasuryManager deployed to:", treasuryAddress);
  console.log();

  // Deploy AgentRegistry
  console.log("🤖 Deploying AgentRegistry...");
  const AgentRegistry = await hre.ethers.getContractFactory("AgentRegistry");
  const agentRegistry = await AgentRegistry.deploy(daoAddress);
  await agentRegistry.waitForDeployment();
  const registryAddress = await agentRegistry.getAddress();
  console.log("✅ AgentRegistry deployed to:", registryAddress);
  console.log();

  // Grant initial voting power to deployer (for testing)
  console.log("🗳️  Granting initial voting power...");
  const tx = await daoGovernance.grantVotingPower(deployer.address, hre.ethers.parseEther("1000"));
  await tx.wait();
  console.log("✅ Granted 1000 voting power to deployer");
  console.log();

  // Register initial agents
  console.log("🤖 Registering AI agents...");

  // Register Momentum Agent
  const agent1Tx = await treasuryManager.registerAgent(
    "Momentum Trader",
    deployer.address,
    3333  // 33.33% allocation
  );
  await agent1Tx.wait();
  console.log("✅ Registered Momentum Trader (Agent 0)");

  // Register Arbitrage Agent
  const agent2Tx = await treasuryManager.registerAgent(
    "Arbitrage Hunter",
    deployer.address,
    3333  // 33.33% allocation
  );
  await agent2Tx.wait();
  console.log("✅ Registered Arbitrage Hunter (Agent 1)");

  // Register Hedging Agent
  const agent3Tx = await treasuryManager.registerAgent(
    "Risk Hedger",
    deployer.address,
    3334  // 33.34% allocation
  );
  await agent3Tx.wait();
  console.log("✅ Registered Risk Hedger (Agent 2)");
  console.log();

  // Save deployment addresses
  const deploymentInfo = {
    network: hre.network.name,
    deployer: deployer.address,
    contracts: {
      DAOGovernance: daoAddress,
      TreasuryManager: treasuryAddress,
      AgentRegistry: registryAddress
    },
    timestamp: new Date().toISOString(),
    blockNumber: await hre.ethers.provider.getBlockNumber()
  };

  const deploymentsDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir);
  }

  const deploymentFile = path.join(deploymentsDir, `${hre.network.name}.json`);
  fs.writeFileSync(deploymentFile, JSON.stringify(deploymentInfo, null, 2));

  console.log("📄 Deployment info saved to:", deploymentFile);
  console.log();

  // Print summary
  console.log("=" .repeat(60));
  console.log("🎉 DEPLOYMENT COMPLETE!");
  console.log("=" .repeat(60));
  console.log("\n📋 Contract Addresses:");
  console.log("  DAOGovernance:   ", daoAddress);
  console.log("  TreasuryManager: ", treasuryAddress);
  console.log("  AgentRegistry:   ", registryAddress);
  console.log("\n🔗 Network:", hre.network.name);
  console.log("📦 Block Number:", await hre.ethers.provider.getBlockNumber());
  console.log();

  if (hre.network.name === "sepolia" || hre.network.name === "mumbai") {
    console.log("🔍 Verify contracts with:");
    console.log(`  npx hardhat verify --network ${hre.network.name} ${daoAddress}`);
    console.log(`  npx hardhat verify --network ${hre.network.name} ${treasuryAddress} ${daoAddress}`);
    console.log(`  npx hardhat verify --network ${hre.network.name} ${registryAddress} ${daoAddress}`);
    console.log();
  }

  console.log("✅ Update config/config.yaml with these addresses");
  console.log("=" .repeat(60));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
