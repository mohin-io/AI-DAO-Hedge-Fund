// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title AgentRegistry
 * @dev Registry and reputation system for AI trading agents
 * @notice Tracks agent metadata, performance history, and staking
 */
contract AgentRegistry {

    struct AgentMetadata {
        string name;
        string strategy;
        string modelHash; // IPFS hash of the model
        address owner;
        uint256 stakedAmount;
        uint256 reputationScore;
        bool isVerified;
        uint256 registeredAt;
    }

    struct PerformanceSnapshot {
        uint256 timestamp;
        int256 pnl;
        uint256 sharpeRatio; // Scaled by 1000
        uint256 maxDrawdown; // In basis points
        uint256 totalTrades;
    }

    // State
    mapping(address => AgentMetadata) public agents;
    mapping(address => PerformanceSnapshot[]) public performanceHistory;
    mapping(address => bool) public isRegistered;

    address[] public registeredAgents;
    address public governance;
    uint256 public minStake = 1 ether;

    // Events
    event AgentRegistered(
        address indexed agentAddress,
        string name,
        string strategy,
        uint256 stakedAmount
    );

    event AgentStaked(address indexed agentAddress, uint256 amount);
    event AgentUnstaked(address indexed agentAddress, uint256 amount);
    event PerformanceRecorded(address indexed agentAddress, int256 pnl, uint256 sharpeRatio);
    event AgentVerified(address indexed agentAddress);
    event ReputationUpdated(address indexed agentAddress, uint256 newScore);

    modifier onlyGovernance() {
        require(msg.sender == governance, "Only governance");
        _;
    }

    constructor(address _governance) {
        governance = _governance;
    }

    /**
     * @dev Register a new agent with stake
     * @param name Agent name
     * @param strategy Strategy description
     * @param modelHash IPFS hash of model weights
     */
    function registerAgent(
        string memory name,
        string memory strategy,
        string memory modelHash
    ) external payable {
        require(!isRegistered[msg.sender], "Already registered");
        require(msg.value >= minStake, "Insufficient stake");

        agents[msg.sender] = AgentMetadata({
            name: name,
            strategy: strategy,
            modelHash: modelHash,
            owner: msg.sender,
            stakedAmount: msg.value,
            reputationScore: 0, // Start at zero
            isVerified: false,
            registeredAt: block.timestamp
        });

        isRegistered[msg.sender] = true;
        registeredAgents.push(msg.sender);

        emit AgentRegistered(msg.sender, name, strategy, msg.value);
    }

    /**
     * @dev Add more stake to an agent
     */
    function stake() external payable {
        require(isRegistered[msg.sender], "Not registered");
        require(msg.value > 0, "Stake must be > 0");

        agents[msg.sender].stakedAmount += msg.value;
        emit AgentStaked(msg.sender, msg.value);
    }

    /**
     * @dev Unstake and withdraw (governance approval required for safety)
     * @param amount Amount to unstake
     */
    function unstake(uint256 amount) external {
        require(isRegistered[msg.sender], "Not registered");
        require(amount > 0, "Amount must be > 0");
        require(agents[msg.sender].stakedAmount >= amount, "Insufficient stake");
        require(
            agents[msg.sender].stakedAmount - amount >= minStake,
            "Must maintain minimum stake"
        );

        agents[msg.sender].stakedAmount -= amount;
        payable(msg.sender).transfer(amount);

        emit AgentUnstaked(msg.sender, amount);
    }

    /**
     * @dev Record agent performance (called by TreasuryManager)
     * @param agentAddress Address of the agent
     * @param pnl Profit/Loss in basis points
     * @param sharpeRatio Sharpe ratio * 1000
     * @param maxDrawdown Max drawdown in basis points
     * @param totalTrades Total trades executed
     */
    function recordPerformance(
        address agentAddress,
        int256 pnl,
        uint256 sharpeRatio,
        uint256 maxDrawdown,
        uint256 totalTrades
    ) external onlyGovernance {
        require(isRegistered[agentAddress], "Agent not registered");

        performanceHistory[agentAddress].push(PerformanceSnapshot({
            timestamp: block.timestamp,
            pnl: pnl,
            sharpeRatio: sharpeRatio,
            maxDrawdown: maxDrawdown,
            totalTrades: totalTrades
        }));

        // Update reputation based on performance
        _updateReputation(agentAddress, pnl, sharpeRatio, maxDrawdown);

        emit PerformanceRecorded(agentAddress, pnl, sharpeRatio);
    }

    /**
     * @dev Internal function to update reputation score
     */
    function _updateReputation(
        address agentAddress,
        int256 pnl,
        uint256 sharpeRatio,
        uint256 maxDrawdown
    ) internal {
        uint256 currentScore = agents[agentAddress].reputationScore;

        // Simple reputation formula:
        // +points for positive PnL and high Sharpe
        // -points for large drawdowns
        int256 reputationDelta = 0;

        if (pnl > 0) {
            reputationDelta += int256(pnl / 100); // Scale down
        } else {
            reputationDelta += pnl / 100;
        }

        if (sharpeRatio > 1500) { // Sharpe > 1.5
            reputationDelta += 10;
        }

        if (maxDrawdown > 2000) { // Drawdown > 20%
            reputationDelta -= 20;
        }

        // Update score with bounds [0, 1000]
        int256 newScore = int256(currentScore) + reputationDelta;
        if (newScore < 0) newScore = 0;
        if (newScore > 1000) newScore = 1000;

        agents[agentAddress].reputationScore = uint256(newScore);
        emit ReputationUpdated(agentAddress, uint256(newScore));
    }

    /**
     * @dev Verify an agent (governance only)
     * @param agentAddress Address to verify
     */
    function verifyAgent(address agentAddress) external onlyGovernance {
        require(isRegistered[agentAddress], "Agent not registered");
        agents[agentAddress].isVerified = true;
        emit AgentVerified(agentAddress);
    }

    /**
     * @dev Get agent full details
     */
    function getAgent(address agentAddress) external view returns (
        string memory name,
        string memory strategy,
        string memory modelHash,
        uint256 stakedAmount,
        uint256 reputationScore,
        bool isVerified,
        uint256 performanceRecords
    ) {
        require(isRegistered[agentAddress], "Agent not registered");
        AgentMetadata memory agent = agents[agentAddress];

        return (
            agent.name,
            agent.strategy,
            agent.modelHash,
            agent.stakedAmount,
            agent.reputationScore,
            agent.isVerified,
            performanceHistory[agentAddress].length
        );
    }

    /**
     * @dev Get performance history
     * @param agentAddress Agent to query
     * @param limit Number of recent records to return
     */
    function getPerformanceHistory(address agentAddress, uint256 limit)
        external
        view
        returns (PerformanceSnapshot[] memory)
    {
        require(isRegistered[agentAddress], "Agent not registered");

        PerformanceSnapshot[] storage history = performanceHistory[agentAddress];
        uint256 length = history.length;

        if (limit > length) {
            limit = length;
        }

        PerformanceSnapshot[] memory recent = new PerformanceSnapshot[](limit);

        for (uint256 i = 0; i < limit; i++) {
            recent[i] = history[length - limit + i];
        }

        return recent;
    }

    /**
     * @dev Get all registered agents
     */
    function getAllAgents() external view returns (address[] memory) {
        return registeredAgents;
    }

    /**
     * @dev Get top performing agents
     * @param count Number of agents to return
     */
    function getTopAgents(uint256 count) external view returns (address[] memory) {
        // If count exceeds registered agents, return all
        if (count > registeredAgents.length) {
            count = registeredAgents.length;
        }

        // Handle empty case
        if (registeredAgents.length == 0) {
            return new address[](0);
        }

        // Simple bubble sort for top reputation scores
        address[] memory sortedAgents = new address[](registeredAgents.length);
        for (uint256 i = 0; i < registeredAgents.length; i++) {
            sortedAgents[i] = registeredAgents[i];
        }

        // Only sort if we have more than 1 agent
        if (sortedAgents.length > 1) {
            for (uint256 i = 0; i < sortedAgents.length - 1; i++) {
                for (uint256 j = 0; j < sortedAgents.length - i - 1; j++) {
                    if (agents[sortedAgents[j]].reputationScore <
                        agents[sortedAgents[j + 1]].reputationScore) {
                        address temp = sortedAgents[j];
                        sortedAgents[j] = sortedAgents[j + 1];
                        sortedAgents[j + 1] = temp;
                    }
                }
            }
        }

        address[] memory topAgents = new address[](count);
        for (uint256 i = 0; i < count; i++) {
            topAgents[i] = sortedAgents[i];
        }

        return topAgents;
    }

    /**
     * @dev Update minimum stake requirement
     */
    function setMinStake(uint256 newMinStake) external onlyGovernance {
        minStake = newMinStake;
    }

    /**
     * @dev Update reputation score manually (governance only)
     * @param agentAddress Address of agent
     * @param newScore New reputation score (0-1000)
     */
    function updateReputation(address agentAddress, uint256 newScore) external onlyGovernance {
        require(isRegistered[agentAddress], "Agent not registered");
        require(newScore <= 1000, "Score must be <= 1000");
        agents[agentAddress].reputationScore = newScore;
        emit ReputationUpdated(agentAddress, newScore);
    }

    /**
     * @dev Get registered agents list
     */
    function getRegisteredAgents() external view returns (address[] memory) {
        return registeredAgents;
    }

    /**
     * @dev Get performance history without limit
     */
    function getPerformanceHistory(address agentAddress) external view returns (PerformanceSnapshot[] memory) {
        return performanceHistory[agentAddress];
    }

    /**
     * @dev Update governance address
     */
    function setGovernance(address newGovernance) external onlyGovernance {
        require(newGovernance != address(0), "Invalid governance address");
        governance = newGovernance;
    }

    /**
     * @dev Update model hash (agent owner only)
     */
    function updateModelHash(string memory newHash) external {
        require(isRegistered[msg.sender], "Not registered");
        require(agents[msg.sender].owner == msg.sender, "Only agent owner");
        agents[msg.sender].modelHash = newHash;
    }
}
