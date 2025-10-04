// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title TreasuryManager
 * @dev Manages the hedge fund's treasury and tracks agent performance
 * @notice Handles deposits, withdrawals, profit distribution, and agent allocations
 */
contract TreasuryManager {

    // Agent structure
    struct Agent {
        string name;
        address agentAddress;
        bool isActive;
        uint256 allocation; // Percentage (0-10000 for 0-100%)
        uint256 totalTrades;
        int256 totalPnL; // Profit and Loss in basis points
        uint256 registeredAt;
    }

    // Investor structure
    struct Investor {
        uint256 shares;
        uint256 depositedAmount;
        uint256 depositTime;
        uint256 lastClaimTime;
    }

    // State variables
    mapping(uint256 => Agent) public agents;
    mapping(address => Investor) public investors;

    uint256 public agentCount;
    uint256 public totalShares;
    uint256 public totalAssets;
    uint256 public performanceFee = 2000; // 20% in basis points
    uint256 public managementFee = 200; // 2% annual in basis points

    address public governance;
    address public admin;
    bool public emergencyStop;

    // Constants
    uint256 constant BASIS_POINTS = 10000;
    uint256 constant SECONDS_PER_YEAR = 365 days;

    // Events
    event AgentRegistered(uint256 indexed agentId, string name, address agentAddress);
    event AgentStatusChanged(uint256 indexed agentId, bool isActive);
    event AllocationUpdated(uint256 indexed agentId, uint256 newAllocation);
    event TradeRecorded(uint256 indexed agentId, int256 pnl, uint256 timestamp);
    event Deposit(address indexed investor, uint256 amount, uint256 shares);
    event Withdrawal(address indexed investor, uint256 shares, uint256 amount);
    event ProfitDistributed(uint256 totalProfit, uint256 performanceFeeAmount);
    event EmergencyStopActivated(uint256 timestamp);

    modifier onlyGovernance() {
        require(msg.sender == governance, "Only governance");
        _;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }

    modifier notStopped() {
        require(!emergencyStop, "Emergency stop active");
        _;
    }

    constructor(address _governance) {
        governance = _governance;
        admin = msg.sender;
        emergencyStop = false;
    }

    /**
     * @dev Register a new AI agent
     * @param name Agent name (e.g., "Momentum Trader")
     * @param agentAddress Address representing the agent
     * @param initialAllocation Initial allocation percentage (0-10000)
     */
    function registerAgent(
        string memory name,
        address agentAddress,
        uint256 initialAllocation
    ) external onlyGovernance returns (uint256) {
        require(initialAllocation <= BASIS_POINTS, "Allocation must be <= 100%");

        agentCount++;
        uint256 agentId = agentCount;

        agents[agentId] = Agent({
            name: name,
            agentAddress: agentAddress,
            isActive: true,
            allocation: initialAllocation,
            totalTrades: 0,
            totalPnL: 0,
            registeredAt: block.timestamp
        });

        emit AgentRegistered(agentId, name, agentAddress);
        emit AllocationUpdated(agentId, initialAllocation);

        return agentId;
    }

    /**
     * @dev Enable or disable an agent
     * @param agentId ID of the agent
     * @param status New status
     */
    function setAgentStatus(uint256 agentId, bool status) external onlyGovernance {
        require(agentId > 0 && agentId <= agentCount, "Agent does not exist");
        agents[agentId].isActive = status;
        emit AgentStatusChanged(agentId, status);
    }

    /**
     * @dev Update agent allocation
     * @param agentId ID of the agent
     * @param newAllocation New allocation percentage
     */
    function updateAllocation(uint256 agentId, uint256 newAllocation)
        external
        onlyGovernance
    {
        require(agentId > 0 && agentId <= agentCount, "Agent does not exist");
        require(newAllocation <= BASIS_POINTS, "Allocation must be <= 100%");

        agents[agentId].allocation = newAllocation;
        emit AllocationUpdated(agentId, newAllocation);
    }

    /**
     * @dev Record a trade executed by an agent
     * @param agentId ID of the agent
     * @param pnl Profit/Loss in basis points (can be negative)
     */
    function recordTrade(uint256 agentId, int256 pnl) external onlyGovernance notStopped {
        require(agentId > 0 && agentId <= agentCount, "Agent does not exist");
        require(agents[agentId].isActive, "Agent is not active");

        agents[agentId].totalTrades++;
        agents[agentId].totalPnL += pnl;

        // Update total assets
        if (pnl > 0) {
            totalAssets += uint256(pnl) * totalAssets / BASIS_POINTS;
        } else if (pnl < 0) {
            totalAssets -= uint256(-pnl) * totalAssets / BASIS_POINTS;
        }

        emit TradeRecorded(agentId, pnl, block.timestamp);
    }

    /**
     * @dev Investor deposits funds
     */
    function deposit() external payable notStopped {
        require(msg.value > 0, "Deposit must be > 0");

        uint256 shares;
        if (totalShares == 0) {
            shares = msg.value;
        } else {
            shares = (msg.value * totalShares) / totalAssets;
        }

        investors[msg.sender].shares += shares;
        investors[msg.sender].depositedAmount += msg.value;
        investors[msg.sender].depositTime = block.timestamp;

        totalShares += shares;
        totalAssets += msg.value;

        emit Deposit(msg.sender, msg.value, shares);
    }

    /**
     * @dev Investor withdraws funds
     * @param shareAmount Amount of shares to redeem
     */
    function withdraw(uint256 shareAmount) external notStopped {
        require(investors[msg.sender].shares >= shareAmount, "Insufficient shares");
        require(shareAmount > 0, "Shares must be > 0");

        uint256 assetAmount = (shareAmount * totalAssets) / totalShares;

        // Deduct management fee
        uint256 timeHeld = block.timestamp - investors[msg.sender].depositTime;
        uint256 managementFeeAmount = (assetAmount * managementFee * timeHeld) /
                                       (BASIS_POINTS * SECONDS_PER_YEAR);

        uint256 netAmount = assetAmount - managementFeeAmount;

        investors[msg.sender].shares -= shareAmount;
        totalShares -= shareAmount;
        totalAssets -= assetAmount;

        payable(msg.sender).transfer(netAmount);

        emit Withdrawal(msg.sender, shareAmount, netAmount);
    }

    /**
     * @dev Distribute profits and deduct performance fee
     */
    function distributeProfits() external onlyAdmin {
        require(totalAssets > 0, "No assets to distribute");

        uint256 totalDeposits = 0;
        // In a real implementation, track total deposits separately

        if (totalAssets > totalDeposits) {
            uint256 profit = totalAssets - totalDeposits;
            uint256 feeAmount = (profit * performanceFee) / BASIS_POINTS;

            totalAssets -= feeAmount;
            payable(admin).transfer(feeAmount);

            emit ProfitDistributed(profit, feeAmount);
        }
    }

    /**
     * @dev Get agent performance metrics
     * @param agentId ID of the agent
     */
    function getAgentPerformance(uint256 agentId) external view returns (
        string memory name,
        bool isActive,
        uint256 allocation,
        uint256 totalTrades,
        int256 totalPnL,
        int256 avgPnLPerTrade
    ) {
        require(agentId > 0 && agentId <= agentCount, "Agent does not exist");
        Agent memory agent = agents[agentId];

        int256 avgPnL = agent.totalTrades > 0 ?
                        agent.totalPnL / int256(agent.totalTrades) :
                        int256(0);

        return (
            agent.name,
            agent.isActive,
            agent.allocation,
            agent.totalTrades,
            agent.totalPnL,
            avgPnL
        );
    }

    /**
     * @dev Get investor position
     * @param investor Address of the investor
     */
    function getInvestorPosition(address investor) external view returns (
        uint256 shares,
        uint256 currentValue,
        uint256 deposited,
        int256 unrealizedPnL
    ) {
        Investor memory inv = investors[investor];
        uint256 value = totalShares > 0 ?
                        (inv.shares * totalAssets) / totalShares :
                        0;
        int256 pnl = int256(value) - int256(inv.depositedAmount);

        return (inv.shares, value, inv.depositedAmount, pnl);
    }

    /**
     * @dev Calculate total allocation across all agents
     */
    function getTotalAllocation() external view returns (uint256) {
        uint256 total = 0;
        for (uint256 i = 1; i <= agentCount; i++) {
            if (agents[i].isActive) {
                total += agents[i].allocation;
            }
        }
        return total;
    }

    /**
     * @dev Emergency stop mechanism
     */
    function activateEmergencyStop() external onlyAdmin {
        emergencyStop = true;
        emit EmergencyStopActivated(block.timestamp);
    }

    /**
     * @dev Update performance fee
     * @param newFee New fee in basis points
     */
    function setPerformanceFee(uint256 newFee) external onlyAdmin {
        require(newFee <= 5000, "Fee must be <= 50%"); // Max 50%
        performanceFee = newFee;
    }

    /**
     * @dev Update management fee
     * @param newFee New fee in basis points
     */
    function setManagementFee(uint256 newFee) external onlyAdmin {
        require(newFee <= 1000, "Fee must be <= 10%"); // Max 10%
        managementFee = newFee;
    }

    /**
     * @dev Get share price (assets per share)
     */
    function getSharePrice() external view returns (uint256) {
        if (totalShares == 0) return 1e18; // 1:1 for first deposit
        return (totalAssets * 1e18) / totalShares;
    }

    /**
     * @dev Get agent PnL
     * @param agentId ID of the agent
     */
    function getAgentPnL(uint256 agentId) external view returns (int256) {
        require(agentId > 0 && agentId <= agentCount, "Agent does not exist");
        return agents[agentId].totalPnL;
    }

    /**
     * @dev Update governance address
     * @param newGovernance New governance contract address
     */
    function setGovernance(address newGovernance) external onlyAdmin {
        require(newGovernance != address(0), "Invalid governance address");
        governance = newGovernance;
    }

    /**
     * @dev Receive function for deposits
     */
    receive() external payable {
        // Deposits handled by deposit() function
        revert("Use deposit() function");
    }
}
