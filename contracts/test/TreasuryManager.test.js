const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("TreasuryManager", function () {
  let treasuryManager, daoGovernance;
  let owner, governance, admin, investor1, investor2, agent1, agent2;
  const PERFORMANCE_FEE = 2000; // 20%
  const MANAGEMENT_FEE = 200; // 2% annual
  const BASIS_POINTS = 10000;

  beforeEach(async function () {
    [owner, governance, admin, investor1, investor2, agent1, agent2] = await ethers.getSigners();

    // Deploy TreasuryManager with governance address
    const TreasuryManager = await ethers.getContractFactory("TreasuryManager");
    treasuryManager = await TreasuryManager.deploy(governance.address);
    await treasuryManager.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the correct governance address", async function () {
      expect(await treasuryManager.governance()).to.equal(governance.address);
    });

    it("Should set the correct admin address", async function () {
      expect(await treasuryManager.admin()).to.equal(owner.address);
    });

    it("Should initialize with emergencyStop = false", async function () {
      expect(await treasuryManager.emergencyStop()).to.equal(false);
    });

    it("Should set correct performance fee", async function () {
      expect(await treasuryManager.performanceFee()).to.equal(PERFORMANCE_FEE);
    });

    it("Should set correct management fee", async function () {
      expect(await treasuryManager.managementFee()).to.equal(MANAGEMENT_FEE);
    });

    it("Should start with zero agents", async function () {
      expect(await treasuryManager.agentCount()).to.equal(0);
    });

    it("Should start with zero total shares", async function () {
      expect(await treasuryManager.totalShares()).to.equal(0);
    });
  });

  describe("Agent Management", function () {
    it("Should allow governance to register new agent", async function () {
      await expect(
        treasuryManager.connect(governance).registerAgent(
          "Momentum Trader",
          agent1.address,
          3000 // 30% allocation
        )
      )
        .to.emit(treasuryManager, "AgentRegistered")
        .withArgs(1, "Momentum Trader", agent1.address);
    });

    it("Should increment agent count", async function () {
      await treasuryManager.connect(governance).registerAgent(
        "Momentum Trader",
        agent1.address,
        3000
      );
      expect(await treasuryManager.agentCount()).to.equal(1);

      await treasuryManager.connect(governance).registerAgent(
        "Arbitrage Trader",
        agent2.address,
        2000
      );
      expect(await treasuryManager.agentCount()).to.equal(2);
    });

    it("Should store agent details correctly", async function () {
      await treasuryManager.connect(governance).registerAgent(
        "Momentum Trader",
        agent1.address,
        3000
      );

      const agent = await treasuryManager.agents(1);
      expect(agent.name).to.equal("Momentum Trader");
      expect(agent.agentAddress).to.equal(agent1.address);
      expect(agent.isActive).to.equal(true);
      expect(agent.allocation).to.equal(3000);
      expect(agent.totalTrades).to.equal(0);
      expect(agent.totalPnL).to.equal(0);
    });

    it("Should revert if non-governance tries to register agent", async function () {
      await expect(
        treasuryManager.connect(investor1).registerAgent(
          "Test Agent",
          agent1.address,
          1000
        )
      ).to.be.revertedWith("Only governance");
    });

    it("Should revert if allocation exceeds 100%", async function () {
      await expect(
        treasuryManager.connect(governance).registerAgent(
          "Test Agent",
          agent1.address,
          15000 // 150%
        )
      ).to.be.revertedWith("Allocation must be <= 100%");
    });
  });

  describe("Agent Status Management", function () {
    beforeEach(async function () {
      await treasuryManager.connect(governance).registerAgent(
        "Momentum Trader",
        agent1.address,
        3000
      );
    });

    it("Should allow governance to deactivate agent", async function () {
      await expect(
        treasuryManager.connect(governance).setAgentStatus(1, false)
      )
        .to.emit(treasuryManager, "AgentStatusChanged")
        .withArgs(1, false);

      const agent = await treasuryManager.agents(1);
      expect(agent.isActive).to.equal(false);
    });

    it("Should allow governance to reactivate agent", async function () {
      await treasuryManager.connect(governance).setAgentStatus(1, false);
      await treasuryManager.connect(governance).setAgentStatus(1, true);

      const agent = await treasuryManager.agents(1);
      expect(agent.isActive).to.equal(true);
    });

    it("Should revert if non-governance tries to change status", async function () {
      await expect(
        treasuryManager.connect(investor1).setAgentStatus(1, false)
      ).to.be.revertedWith("Only governance");
    });

    it("Should revert for non-existent agent", async function () {
      await expect(
        treasuryManager.connect(governance).setAgentStatus(999, false)
      ).to.be.revertedWith("Agent does not exist");
    });
  });

  describe("Allocation Management", function () {
    beforeEach(async function () {
      await treasuryManager.connect(governance).registerAgent(
        "Momentum Trader",
        agent1.address,
        3000
      );
    });

    it("Should allow governance to update allocation", async function () {
      await expect(
        treasuryManager.connect(governance).updateAllocation(1, 5000)
      )
        .to.emit(treasuryManager, "AllocationUpdated")
        .withArgs(1, 5000);

      const agent = await treasuryManager.agents(1);
      expect(agent.allocation).to.equal(5000);
    });

    it("Should revert if allocation exceeds 100%", async function () {
      await expect(
        treasuryManager.connect(governance).updateAllocation(1, 12000)
      ).to.be.revertedWith("Allocation must be <= 100%");
    });

    it("Should revert if non-governance tries to update", async function () {
      await expect(
        treasuryManager.connect(investor1).updateAllocation(1, 4000)
      ).to.be.revertedWith("Only governance");
    });
  });

  describe("Deposits", function () {
    it("Should allow investors to deposit funds", async function () {
      const depositAmount = ethers.parseEther("10");

      await expect(
        treasuryManager.connect(investor1).deposit({ value: depositAmount })
      )
        .to.emit(treasuryManager, "Deposit")
        .withArgs(investor1.address, depositAmount, depositAmount); // 1:1 for first deposit
    });

    it("Should update investor shares correctly", async function () {
      const depositAmount = ethers.parseEther("10");
      await treasuryManager.connect(investor1).deposit({ value: depositAmount });

      const investor = await treasuryManager.investors(investor1.address);
      expect(investor.shares).to.equal(depositAmount);
      expect(investor.depositedAmount).to.equal(depositAmount);
    });

    it("Should update total shares and assets", async function () {
      const depositAmount = ethers.parseEther("10");
      await treasuryManager.connect(investor1).deposit({ value: depositAmount });

      expect(await treasuryManager.totalShares()).to.equal(depositAmount);
      expect(await treasuryManager.totalAssets()).to.equal(depositAmount);
    });

    it("Should handle multiple deposits correctly", async function () {
      const amount1 = ethers.parseEther("10");
      const amount2 = ethers.parseEther("5");

      await treasuryManager.connect(investor1).deposit({ value: amount1 });
      await treasuryManager.connect(investor2).deposit({ value: amount2 });

      const totalExpected = amount1 + amount2;
      expect(await treasuryManager.totalAssets()).to.equal(totalExpected);
    });

    it("Should revert deposit of zero amount", async function () {
      await expect(
        treasuryManager.connect(investor1).deposit({ value: 0 })
      ).to.be.revertedWith("Deposit must be > 0");
    });

    it("Should revert when emergency stop is active", async function () {
      await treasuryManager.connect(owner).activateEmergencyStop();

      await expect(
        treasuryManager.connect(investor1).deposit({ value: ethers.parseEther("10") })
      ).to.be.revertedWith("Emergency stop active");
    });
  });

  describe("Withdrawals", function () {
    beforeEach(async function () {
      // Investor deposits funds
      await treasuryManager.connect(investor1).deposit({ value: ethers.parseEther("10") });
    });

    it("Should allow investors to withdraw their shares", async function () {
      const investor = await treasuryManager.investors(investor1.address);
      const sharesToWithdraw = investor.shares / BigInt(2); // Withdraw half

      await expect(
        treasuryManager.connect(investor1).withdraw(sharesToWithdraw)
      ).to.emit(treasuryManager, "Withdrawal");
    });

    it("Should update investor shares after withdrawal", async function () {
      const investor = await treasuryManager.investors(investor1.address);
      const initialShares = investor.shares;
      const sharesToWithdraw = initialShares / BigInt(2);

      await treasuryManager.connect(investor1).withdraw(sharesToWithdraw);

      const updatedInvestor = await treasuryManager.investors(investor1.address);
      expect(updatedInvestor.shares).to.equal(initialShares - sharesToWithdraw);
    });

    it("Should revert if withdrawing more shares than owned", async function () {
      const investor = await treasuryManager.investors(investor1.address);
      const excessiveAmount = investor.shares + BigInt(1);

      await expect(
        treasuryManager.connect(investor1).withdraw(excessiveAmount)
      ).to.be.revertedWith("Insufficient shares");
    });

    it("Should revert withdrawal of zero shares", async function () {
      await expect(
        treasuryManager.connect(investor1).withdraw(0)
      ).to.be.revertedWith("Shares must be > 0");
    });

    it("Should revert when emergency stop is active", async function () {
      await treasuryManager.connect(owner).activateEmergencyStop();

      const investor = await treasuryManager.investors(investor1.address);
      await expect(
        treasuryManager.connect(investor1).withdraw(investor.shares)
      ).to.be.revertedWith("Emergency stop active");
    });
  });

  describe("Trade Recording", function () {
    beforeEach(async function () {
      await treasuryManager.connect(governance).registerAgent(
        "Momentum Trader",
        agent1.address,
        3000
      );
    });

    it("Should allow governance to record profitable trade", async function () {
      const pnl = 1000; // Positive PnL

      await expect(
        treasuryManager.connect(governance).recordTrade(1, pnl)
      ).to.emit(treasuryManager, "TradeRecorded");
    });

    it("Should allow governance to record losing trade", async function () {
      const pnl = -500; // Negative PnL

      await expect(
        treasuryManager.connect(governance).recordTrade(1, pnl)
      ).to.emit(treasuryManager, "TradeRecorded");
    });

    it("Should update agent statistics correctly", async function () {
      await treasuryManager.connect(governance).recordTrade(1, 1000);
      await treasuryManager.connect(governance).recordTrade(1, -500);
      await treasuryManager.connect(governance).recordTrade(1, 750);

      const agent = await treasuryManager.agents(1);
      expect(agent.totalTrades).to.equal(3);
      expect(agent.totalPnL).to.equal(1250); // 1000 - 500 + 750
    });

    it("Should revert if non-governance tries to record trade", async function () {
      await expect(
        treasuryManager.connect(investor1).recordTrade(1, 1000)
      ).to.be.revertedWith("Only governance");
    });

    it("Should revert for non-existent agent", async function () {
      await expect(
        treasuryManager.connect(governance).recordTrade(999, 1000)
      ).to.be.revertedWith("Agent does not exist");
    });

    it("Should revert for inactive agent", async function () {
      await treasuryManager.connect(governance).setAgentStatus(1, false);

      await expect(
        treasuryManager.connect(governance).recordTrade(1, 1000)
      ).to.be.revertedWith("Agent is not active");
    });
  });

  describe("Performance Metrics", function () {
    beforeEach(async function () {
      await treasuryManager.connect(governance).registerAgent(
        "Momentum Trader",
        agent1.address,
        3000
      );
      await treasuryManager.connect(investor1).deposit({ value: ethers.parseEther("10") });
    });

    it("Should calculate share price correctly", async function () {
      const sharePrice = await treasuryManager.getSharePrice();
      expect(sharePrice).to.be.gt(0);
    });

    it("Should calculate agent PnL correctly", async function () {
      await treasuryManager.connect(governance).recordTrade(1, 1000);
      await treasuryManager.connect(governance).recordTrade(1, -200);

      const pnl = await treasuryManager.getAgentPnL(1);
      expect(pnl).to.equal(800);
    });

    it("Should return zero PnL for agent with no trades", async function () {
      const pnl = await treasuryManager.getAgentPnL(1);
      expect(pnl).to.equal(0);
    });
  });

  describe("Fee Management", function () {
    it("Should allow admin to update performance fee", async function () {
      const newFee = 1500; // 15%
      await treasuryManager.connect(owner).setPerformanceFee(newFee);
      expect(await treasuryManager.performanceFee()).to.equal(newFee);
    });

    it("Should allow admin to update management fee", async function () {
      const newFee = 150; // 1.5%
      await treasuryManager.connect(owner).setManagementFee(newFee);
      expect(await treasuryManager.managementFee()).to.equal(newFee);
    });

    it("Should revert if performance fee exceeds limit", async function () {
      await expect(
        treasuryManager.connect(owner).setPerformanceFee(5001) // > 50%
      ).to.be.revertedWith("Fee must be <= 50%");
    });

    it("Should revert if management fee exceeds limit", async function () {
      await expect(
        treasuryManager.connect(owner).setManagementFee(1001) // > 10%
      ).to.be.revertedWith("Fee must be <= 10%");
    });

    it("Should revert if non-admin tries to update fees", async function () {
      await expect(
        treasuryManager.connect(investor1).setPerformanceFee(1000)
      ).to.be.revertedWith("Only admin");
    });
  });

  describe("Emergency Stop", function () {
    it("Should allow admin to activate emergency stop", async function () {
      await expect(
        treasuryManager.connect(owner).activateEmergencyStop()
      ).to.emit(treasuryManager, "EmergencyStopActivated");

      expect(await treasuryManager.emergencyStop()).to.equal(true);
    });

    it("Should prevent deposits when emergency stop is active", async function () {
      await treasuryManager.connect(owner).activateEmergencyStop();

      await expect(
        treasuryManager.connect(investor1).deposit({ value: ethers.parseEther("1") })
      ).to.be.revertedWith("Emergency stop active");
    });

    it("Should prevent withdrawals when emergency stop is active", async function () {
      await treasuryManager.connect(investor1).deposit({ value: ethers.parseEther("10") });
      await treasuryManager.connect(owner).activateEmergencyStop();

      await expect(
        treasuryManager.connect(investor1).withdraw(ethers.parseEther("1"))
      ).to.be.revertedWith("Emergency stop active");
    });

    it("Should revert if non-admin tries to activate emergency stop", async function () {
      await expect(
        treasuryManager.connect(investor1).activateEmergencyStop()
      ).to.be.revertedWith("Only admin");
    });
  });

  describe("Governance Updates", function () {
    it("Should allow admin to update governance address", async function () {
      const newGovernance = investor1.address;
      await treasuryManager.connect(owner).setGovernance(newGovernance);
      expect(await treasuryManager.governance()).to.equal(newGovernance);
    });

    it("Should revert if non-admin tries to update governance", async function () {
      await expect(
        treasuryManager.connect(investor1).setGovernance(investor2.address)
      ).to.be.revertedWith("Only admin");
    });

    it("Should revert if setting governance to zero address", async function () {
      await expect(
        treasuryManager.connect(owner).setGovernance(ethers.ZeroAddress)
      ).to.be.revertedWith("Invalid governance address");
    });
  });
});
