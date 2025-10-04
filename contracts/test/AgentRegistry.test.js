const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("AgentRegistry", function () {
  let agentRegistry;
  let owner, governance, agent1, agent2, agent3;
  const MIN_STAKE = ethers.parseEther("1");

  beforeEach(async function () {
    [owner, governance, agent1, agent2, agent3] = await ethers.getSigners();

    // Deploy AgentRegistry
    const AgentRegistry = await ethers.getContractFactory("AgentRegistry");
    agentRegistry = await AgentRegistry.deploy(governance.address);
    await agentRegistry.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the correct governance address", async function () {
      expect(await agentRegistry.governance()).to.equal(governance.address);
    });

    it("Should set correct minimum stake", async function () {
      expect(await agentRegistry.minStake()).to.equal(MIN_STAKE);
    });

    it("Should start with no registered agents", async function () {
      const agents = await agentRegistry.getRegisteredAgents();
      expect(agents.length).to.equal(0);
    });
  });

  describe("Agent Registration", function () {
    const agentName = "Momentum Trader";
    const strategy = "PPO-based momentum trading";
    const modelHash = "QmX1234..."; // IPFS hash

    it("Should allow agent registration with sufficient stake", async function () {
      await expect(
        agentRegistry.connect(agent1).registerAgent(
          agentName,
          strategy,
          modelHash,
          { value: MIN_STAKE }
        )
      )
        .to.emit(agentRegistry, "AgentRegistered")
        .withArgs(agent1.address, agentName, strategy, MIN_STAKE);
    });

    it("Should store agent metadata correctly", async function () {
      await agentRegistry.connect(agent1).registerAgent(
        agentName,
        strategy,
        modelHash,
        { value: MIN_STAKE }
      );

      const agent = await agentRegistry.agents(agent1.address);
      expect(agent.name).to.equal(agentName);
      expect(agent.strategy).to.equal(strategy);
      expect(agent.modelHash).to.equal(modelHash);
      expect(agent.owner).to.equal(agent1.address);
      expect(agent.stakedAmount).to.equal(MIN_STAKE);
      expect(agent.reputationScore).to.equal(0);
      expect(agent.isVerified).to.equal(false);
    });

    it("Should mark agent as registered", async function () {
      await agentRegistry.connect(agent1).registerAgent(
        agentName,
        strategy,
        modelHash,
        { value: MIN_STAKE }
      );

      expect(await agentRegistry.isRegistered(agent1.address)).to.equal(true);
    });

    it("Should add agent to registered agents list", async function () {
      await agentRegistry.connect(agent1).registerAgent(
        agentName,
        strategy,
        modelHash,
        { value: MIN_STAKE }
      );

      const registeredAgents = await agentRegistry.getRegisteredAgents();
      expect(registeredAgents).to.include(agent1.address);
    });

    it("Should revert if agent is already registered", async function () {
      await agentRegistry.connect(agent1).registerAgent(
        agentName,
        strategy,
        modelHash,
        { value: MIN_STAKE }
      );

      await expect(
        agentRegistry.connect(agent1).registerAgent(
          "Another Name",
          "Another Strategy",
          "QmY5678...",
          { value: MIN_STAKE }
        )
      ).to.be.revertedWith("Already registered");
    });

    it("Should revert if stake is insufficient", async function () {
      const insufficientStake = MIN_STAKE - BigInt(1);

      await expect(
        agentRegistry.connect(agent1).registerAgent(
          agentName,
          strategy,
          modelHash,
          { value: insufficientStake }
        )
      ).to.be.revertedWith("Insufficient stake");
    });

    it("Should accept stake greater than minimum", async function () {
      const extraStake = MIN_STAKE * BigInt(2);

      await agentRegistry.connect(agent1).registerAgent(
        agentName,
        strategy,
        modelHash,
        { value: extraStake }
      );

      const agent = await agentRegistry.agents(agent1.address);
      expect(agent.stakedAmount).to.equal(extraStake);
    });
  });

  describe("Additional Staking", function () {
    beforeEach(async function () {
      await agentRegistry.connect(agent1).registerAgent(
        "Momentum Trader",
        "PPO-based",
        "QmHash",
        { value: MIN_STAKE }
      );
    });

    it("Should allow agents to stake additional funds", async function () {
      const additionalStake = ethers.parseEther("0.5");

      await expect(
        agentRegistry.connect(agent1).stake({ value: additionalStake })
      )
        .to.emit(agentRegistry, "AgentStaked")
        .withArgs(agent1.address, additionalStake);
    });

    it("Should update staked amount correctly", async function () {
      const additionalStake = ethers.parseEther("0.5");
      await agentRegistry.connect(agent1).stake({ value: additionalStake });

      const agent = await agentRegistry.agents(agent1.address);
      expect(agent.stakedAmount).to.equal(MIN_STAKE + additionalStake);
    });

    it("Should revert if non-registered agent tries to stake", async function () {
      await expect(
        agentRegistry.connect(agent2).stake({ value: ethers.parseEther("1") })
      ).to.be.revertedWith("Not registered");
    });

    it("Should revert if staking zero amount", async function () {
      await expect(
        agentRegistry.connect(agent1).stake({ value: 0 })
      ).to.be.revertedWith("Stake must be > 0");
    });
  });

  describe("Unstaking", function () {
    beforeEach(async function () {
      await agentRegistry.connect(agent1).registerAgent(
        "Momentum Trader",
        "PPO-based",
        "QmHash",
        { value: MIN_STAKE * BigInt(3) } // 3 ETH staked
      );
    });

    it("Should allow agents to unstake funds", async function () {
      const unstakeAmount = MIN_STAKE;

      await expect(
        agentRegistry.connect(agent1).unstake(unstakeAmount)
      )
        .to.emit(agentRegistry, "AgentUnstaked")
        .withArgs(agent1.address, unstakeAmount);
    });

    it("Should update staked amount correctly after unstaking", async function () {
      const unstakeAmount = MIN_STAKE;
      const initialStake = MIN_STAKE * BigInt(3);

      await agentRegistry.connect(agent1).unstake(unstakeAmount);

      const agent = await agentRegistry.agents(agent1.address);
      expect(agent.stakedAmount).to.equal(initialStake - unstakeAmount);
    });

    it("Should transfer funds back to agent", async function () {
      const unstakeAmount = MIN_STAKE;
      const initialBalance = await ethers.provider.getBalance(agent1.address);

      const tx = await agentRegistry.connect(agent1).unstake(unstakeAmount);
      const receipt = await tx.wait();
      const gasCost = receipt.gasUsed * tx.gasPrice;

      const finalBalance = await ethers.provider.getBalance(agent1.address);
      expect(finalBalance).to.equal(initialBalance + unstakeAmount - gasCost);
    });

    it("Should revert if unstaking would go below minimum", async function () {
      const excessiveUnstake = MIN_STAKE * BigInt(3); // Would leave 0

      await expect(
        agentRegistry.connect(agent1).unstake(excessiveUnstake)
      ).to.be.revertedWith("Must maintain minimum stake");
    });

    it("Should revert if non-registered agent tries to unstake", async function () {
      await expect(
        agentRegistry.connect(agent2).unstake(MIN_STAKE)
      ).to.be.revertedWith("Not registered");
    });

    it("Should revert if unstaking zero amount", async function () {
      await expect(
        agentRegistry.connect(agent1).unstake(0)
      ).to.be.revertedWith("Amount must be > 0");
    });
  });

  describe("Performance Recording", function () {
    beforeEach(async function () {
      await agentRegistry.connect(agent1).registerAgent(
        "Momentum Trader",
        "PPO-based",
        "QmHash",
        { value: MIN_STAKE }
      );
    });

    it("Should allow governance to record performance", async function () {
      const pnl = 1000;
      const sharpeRatio = 2100; // 2.1 * 1000
      const maxDrawdown = 1500; // 15% in basis points
      const totalTrades = 50;

      await expect(
        agentRegistry.connect(governance).recordPerformance(
          agent1.address,
          pnl,
          sharpeRatio,
          maxDrawdown,
          totalTrades
        )
      )
        .to.emit(agentRegistry, "PerformanceRecorded")
        .withArgs(agent1.address, pnl, sharpeRatio);
    });

    it("Should store performance snapshot correctly", async function () {
      await agentRegistry.connect(governance).recordPerformance(
        agent1.address,
        1000,
        2100,
        1500,
        50
      );

      const history = await agentRegistry.getPerformanceHistory(agent1.address);
      expect(history.length).to.equal(1);
      expect(history[0].pnl).to.equal(1000);
      expect(history[0].sharpeRatio).to.equal(2100);
      expect(history[0].maxDrawdown).to.equal(1500);
      expect(history[0].totalTrades).to.equal(50);
    });

    it("Should maintain performance history over multiple recordings", async function () {
      await agentRegistry.connect(governance).recordPerformance(
        agent1.address,
        1000,
        2100,
        1500,
        50
      );

      await agentRegistry.connect(governance).recordPerformance(
        agent1.address,
        1500,
        2300,
        1200,
        75
      );

      const history = await agentRegistry.getPerformanceHistory(agent1.address);
      expect(history.length).to.equal(2);
    });

    it("Should revert if non-governance tries to record performance", async function () {
      await expect(
        agentRegistry.connect(agent1).recordPerformance(
          agent1.address,
          1000,
          2100,
          1500,
          50
        )
      ).to.be.revertedWith("Only governance");
    });

    it("Should revert for non-registered agent", async function () {
      await expect(
        agentRegistry.connect(governance).recordPerformance(
          agent2.address,
          1000,
          2100,
          1500,
          50
        )
      ).to.be.revertedWith("Agent not registered");
    });
  });

  describe("Agent Verification", function () {
    beforeEach(async function () {
      await agentRegistry.connect(agent1).registerAgent(
        "Momentum Trader",
        "PPO-based",
        "QmHash",
        { value: MIN_STAKE }
      );
    });

    it("Should allow governance to verify agent", async function () {
      await expect(
        agentRegistry.connect(governance).verifyAgent(agent1.address)
      )
        .to.emit(agentRegistry, "AgentVerified")
        .withArgs(agent1.address);
    });

    it("Should update agent verification status", async function () {
      await agentRegistry.connect(governance).verifyAgent(agent1.address);

      const agent = await agentRegistry.agents(agent1.address);
      expect(agent.isVerified).to.equal(true);
    });

    it("Should revert if non-governance tries to verify", async function () {
      await expect(
        agentRegistry.connect(agent1).verifyAgent(agent1.address)
      ).to.be.revertedWith("Only governance");
    });

    it("Should revert for non-registered agent", async function () {
      await expect(
        agentRegistry.connect(governance).verifyAgent(agent2.address)
      ).to.be.revertedWith("Agent not registered");
    });

    it("Should allow verification of already verified agent (idempotent)", async function () {
      await agentRegistry.connect(governance).verifyAgent(agent1.address);
      await expect(
        agentRegistry.connect(governance).verifyAgent(agent1.address)
      ).to.not.be.reverted;
    });
  });

  describe("Reputation System", function () {
    beforeEach(async function () {
      await agentRegistry.connect(agent1).registerAgent(
        "Momentum Trader",
        "PPO-based",
        "QmHash",
        { value: MIN_STAKE }
      );
    });

    it("Should allow governance to update reputation", async function () {
      const newScore = 850;

      await expect(
        agentRegistry.connect(governance).updateReputation(agent1.address, newScore)
      )
        .to.emit(agentRegistry, "ReputationUpdated")
        .withArgs(agent1.address, newScore);
    });

    it("Should update reputation score correctly", async function () {
      await agentRegistry.connect(governance).updateReputation(agent1.address, 850);

      const agent = await agentRegistry.agents(agent1.address);
      expect(agent.reputationScore).to.equal(850);
    });

    it("Should allow reputation to be updated multiple times", async function () {
      await agentRegistry.connect(governance).updateReputation(agent1.address, 750);
      await agentRegistry.connect(governance).updateReputation(agent1.address, 900);

      const agent = await agentRegistry.agents(agent1.address);
      expect(agent.reputationScore).to.equal(900);
    });

    it("Should revert if non-governance tries to update reputation", async function () {
      await expect(
        agentRegistry.connect(agent1).updateReputation(agent1.address, 850)
      ).to.be.revertedWith("Only governance");
    });

    it("Should revert if reputation exceeds maximum (1000)", async function () {
      await expect(
        agentRegistry.connect(governance).updateReputation(agent1.address, 1001)
      ).to.be.revertedWith("Score must be <= 1000");
    });
  });

  describe("Query Functions", function () {
    beforeEach(async function () {
      // Register multiple agents
      await agentRegistry.connect(agent1).registerAgent(
        "Momentum Trader",
        "PPO-based",
        "QmHash1",
        { value: MIN_STAKE }
      );

      await agentRegistry.connect(agent2).registerAgent(
        "Arbitrage Trader",
        "DQN-based",
        "QmHash2",
        { value: MIN_STAKE * BigInt(2) }
      );

      await agentRegistry.connect(agent3).registerAgent(
        "Hedging Agent",
        "SAC-based",
        "QmHash3",
        { value: MIN_STAKE * BigInt(3) }
      );

      // Record performance
      await agentRegistry.connect(governance).recordPerformance(
        agent1.address,
        1000,
        2100,
        1500,
        50
      );

      // Set reputations
      await agentRegistry.connect(governance).updateReputation(agent1.address, 800);
      await agentRegistry.connect(governance).updateReputation(agent2.address, 950);
      await agentRegistry.connect(governance).updateReputation(agent3.address, 700);
    });

    it("Should return all registered agents", async function () {
      const agents = await agentRegistry.getRegisteredAgents();
      expect(agents.length).to.equal(3);
      expect(agents).to.include(agent1.address);
      expect(agents).to.include(agent2.address);
      expect(agents).to.include(agent3.address);
    });

    it("Should return performance history for agent", async function () {
      const history = await agentRegistry.getPerformanceHistory(agent1.address);
      expect(history.length).to.equal(1);
      expect(history[0].pnl).to.equal(1000);
    });

    it("Should return empty history for agent with no performance", async function () {
      const history = await agentRegistry.getPerformanceHistory(agent2.address);
      expect(history.length).to.equal(0);
    });

    it("Should return top agents by reputation", async function () {
      const topAgents = await agentRegistry.getTopAgents(2);
      expect(topAgents.length).to.equal(2);
      expect(topAgents[0]).to.equal(agent2.address); // 950 reputation
      expect(topAgents[1]).to.equal(agent1.address); // 800 reputation
    });

    it("Should return all agents if requesting more than exist", async function () {
      const topAgents = await agentRegistry.getTopAgents(10);
      expect(topAgents.length).to.equal(3);
    });
  });

  describe("Governance Updates", function () {
    it("Should allow current governance to update governance address", async function () {
      const newGovernance = agent1.address;
      await agentRegistry.connect(governance).setGovernance(newGovernance);
      expect(await agentRegistry.governance()).to.equal(newGovernance);
    });

    it("Should allow governance to update minimum stake", async function () {
      const newMinStake = ethers.parseEther("2");
      await agentRegistry.connect(governance).setMinStake(newMinStake);
      expect(await agentRegistry.minStake()).to.equal(newMinStake);
    });

    it("Should revert if non-governance tries to update governance", async function () {
      await expect(
        agentRegistry.connect(agent1).setGovernance(agent2.address)
      ).to.be.revertedWith("Only governance");
    });

    it("Should revert if non-governance tries to update min stake", async function () {
      await expect(
        agentRegistry.connect(agent1).setMinStake(ethers.parseEther("2"))
      ).to.be.revertedWith("Only governance");
    });

    it("Should revert if setting governance to zero address", async function () {
      await expect(
        agentRegistry.connect(governance).setGovernance(ethers.ZeroAddress)
      ).to.be.revertedWith("Invalid governance address");
    });
  });

  describe("Model Hash Updates", function () {
    beforeEach(async function () {
      await agentRegistry.connect(agent1).registerAgent(
        "Momentum Trader",
        "PPO-based",
        "QmHash1",
        { value: MIN_STAKE }
      );
    });

    it("Should allow agent owner to update model hash", async function () {
      const newHash = "QmNewHash";
      await agentRegistry.connect(agent1).updateModelHash(newHash);

      const agent = await agentRegistry.agents(agent1.address);
      expect(agent.modelHash).to.equal(newHash);
    });

    it("Should revert if non-owner tries to update model hash", async function () {
      await expect(
        agentRegistry.connect(agent2).updateModelHash("QmNewHash")
      ).to.be.revertedWith("Not registered");
    });

    it("Should revert for non-registered agent", async function () {
      await expect(
        agentRegistry.connect(agent2).updateModelHash("QmNewHash")
      ).to.be.revertedWith("Not registered");
    });
  });
});
