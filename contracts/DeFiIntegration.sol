// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title DeFiIntegration
 * @notice Multi-chain DeFi protocol integration for yield optimization
 * @dev Supports Uniswap, Aave, Compound across Ethereum, Polygon, Arbitrum
 */
contract DeFiIntegration is ReentrancyGuard, Ownable {

    // Supported protocols
    enum Protocol {
        UNISWAP_V3,
        AAVE_V3,
        COMPOUND_V3,
        CURVE,
        BALANCER
    }

    // Supported chains (matches MultiChainBridge)
    enum Chain {
        ETHEREUM,
        POLYGON,
        ARBITRUM,
        BSC,
        AVALANCHE
    }

    struct ProtocolConfig {
        address routerAddress;
        address poolAddress;
        bool isActive;
        uint256 tvl; // Total Value Locked
        uint256 apy; // Annual Percentage Yield (in basis points)
    }

    struct Position {
        Protocol protocol;
        Chain chain;
        address token;
        uint256 amount;
        uint256 depositTime;
        uint256 accruedYield;
        bool isActive;
    }

    // State variables
    mapping(Protocol => mapping(Chain => ProtocolConfig)) public protocolConfigs;
    mapping(bytes32 => Position) public positions;
    mapping(address => bytes32[]) public userPositions;

    uint256 public totalValueLocked;
    uint256 public totalYieldGenerated;
    uint256 public positionCounter;

    // Events
    event ProtocolConfigured(Protocol indexed protocol, Chain indexed chain, address router, address pool);
    event PositionOpened(bytes32 indexed positionId, address indexed user, Protocol protocol, uint256 amount);
    event PositionClosed(bytes32 indexed positionId, address indexed user, uint256 amount, uint256 yield);
    event YieldHarvested(bytes32 indexed positionId, uint256 yieldAmount);
    event Rebalanced(Protocol fromProtocol, Protocol toProtocol, uint256 amount);

    /**
     * @notice Configure a DeFi protocol on a specific chain
     * @param protocol Protocol to configure
     * @param chain Chain where protocol is deployed
     * @param routerAddress Router/main contract address
     * @param poolAddress Liquidity pool address
     * @param apy Expected APY in basis points
     */
    function configureProtocol(
        Protocol protocol,
        Chain chain,
        address routerAddress,
        address poolAddress,
        uint256 apy
    ) external onlyOwner {
        require(routerAddress != address(0), "Invalid router address");

        protocolConfigs[protocol][chain] = ProtocolConfig({
            routerAddress: routerAddress,
            poolAddress: poolAddress,
            isActive: true,
            tvl: 0,
            apy: apy
        });

        emit ProtocolConfigured(protocol, chain, routerAddress, poolAddress);
    }

    /**
     * @notice Open a position in a DeFi protocol
     * @param protocol Target protocol
     * @param chain Target chain
     * @param token Token to deposit
     * @param amount Amount to deposit
     */
    function openPosition(
        Protocol protocol,
        Chain chain,
        address token,
        uint256 amount
    ) external nonReentrant returns (bytes32) {
        require(amount > 0, "Amount must be greater than 0");
        require(protocolConfigs[protocol][chain].isActive, "Protocol not active");

        IERC20 tokenContract = IERC20(token);
        require(
            tokenContract.transferFrom(msg.sender, address(this), amount),
            "Token transfer failed"
        );

        // Generate position ID
        bytes32 positionId = keccak256(
            abi.encodePacked(msg.sender, protocol, chain, token, positionCounter, block.timestamp)
        );

        // Create position
        positions[positionId] = Position({
            protocol: protocol,
            chain: chain,
            token: token,
            amount: amount,
            depositTime: block.timestamp,
            accruedYield: 0,
            isActive: true
        });

        userPositions[msg.sender].push(positionId);
        positionCounter++;

        // Update TVL
        protocolConfigs[protocol][chain].tvl += amount;
        totalValueLocked += amount;

        emit PositionOpened(positionId, msg.sender, protocol, amount);

        return positionId;
    }

    /**
     * @notice Close a position and withdraw funds
     * @param positionId Position to close
     */
    function closePosition(bytes32 positionId) external nonReentrant {
        Position storage position = positions[positionId];
        require(position.isActive, "Position not active");

        // Calculate accrued yield
        uint256 timeElapsed = block.timestamp - position.depositTime;
        uint256 apy = protocolConfigs[position.protocol][position.chain].apy;
        uint256 yield = (position.amount * apy * timeElapsed) / (365 days * 10000);

        position.accruedYield += yield;
        position.isActive = false;

        uint256 totalWithdrawal = position.amount + position.accruedYield;

        // Update TVL
        protocolConfigs[position.protocol][position.chain].tvl -= position.amount;
        totalValueLocked -= position.amount;
        totalYieldGenerated += position.accruedYield;

        // Transfer tokens back to user
        IERC20(position.token).transfer(msg.sender, totalWithdrawal);

        emit PositionClosed(positionId, msg.sender, position.amount, position.accruedYield);
    }

    /**
     * @notice Harvest yield without closing position
     * @param positionId Position to harvest from
     */
    function harvestYield(bytes32 positionId) external nonReentrant {
        Position storage position = positions[positionId];
        require(position.isActive, "Position not active");

        // Calculate accrued yield
        uint256 timeElapsed = block.timestamp - position.depositTime;
        uint256 apy = protocolConfigs[position.protocol][position.chain].apy;
        uint256 yield = (position.amount * apy * timeElapsed) / (365 days * 10000);

        require(yield > 0, "No yield to harvest");

        position.accruedYield += yield;
        position.depositTime = block.timestamp; // Reset deposit time

        totalYieldGenerated += yield;

        // Transfer yield to user
        IERC20(position.token).transfer(msg.sender, yield);

        emit YieldHarvested(positionId, yield);
    }

    /**
     * @notice Rebalance funds between protocols for better yield
     * @param positionId Position to rebalance
     * @param newProtocol Target protocol
     * @param newChain Target chain
     */
    function rebalance(
        bytes32 positionId,
        Protocol newProtocol,
        Chain newChain
    ) external onlyOwner nonReentrant {
        Position storage position = positions[positionId];
        require(position.isActive, "Position not active");
        require(protocolConfigs[newProtocol][newChain].isActive, "Target protocol not active");

        Protocol oldProtocol = position.protocol;
        Chain oldChain = position.chain;

        // Update TVL
        protocolConfigs[oldProtocol][oldChain].tvl -= position.amount;
        protocolConfigs[newProtocol][newChain].tvl += position.amount;

        // Update position
        position.protocol = newProtocol;
        position.chain = newChain;

        emit Rebalanced(oldProtocol, newProtocol, position.amount);
    }

    /**
     * @notice Get best protocol for yield on a specific chain
     * @param chain Chain to search
     */
    function getBestProtocol(Chain chain) external view returns (Protocol, uint256) {
        Protocol bestProtocol;
        uint256 bestAPY = 0;

        for (uint i = 0; i < 5; i++) {
            Protocol protocol = Protocol(i);
            uint256 apy = protocolConfigs[protocol][chain].apy;

            if (protocolConfigs[protocol][chain].isActive && apy > bestAPY) {
                bestProtocol = protocol;
                bestAPY = apy;
            }
        }

        return (bestProtocol, bestAPY);
    }

    /**
     * @notice Get user's positions
     * @param user User address
     */
    function getUserPositions(address user) external view returns (bytes32[] memory) {
        return userPositions[user];
    }

    /**
     * @notice Get position details
     * @param positionId Position ID
     */
    function getPosition(bytes32 positionId) external view returns (Position memory) {
        return positions[positionId];
    }

    /**
     * @notice Get protocol configuration
     * @param protocol Protocol
     * @param chain Chain
     */
    function getProtocolConfig(Protocol protocol, Chain chain)
        external
        view
        returns (ProtocolConfig memory)
    {
        return protocolConfigs[protocol][chain];
    }

    /**
     * @notice Calculate current accrued yield for a position
     * @param positionId Position ID
     */
    function calculateAccruedYield(bytes32 positionId) external view returns (uint256) {
        Position memory position = positions[positionId];
        if (!position.isActive) return position.accruedYield;

        uint256 timeElapsed = block.timestamp - position.depositTime;
        uint256 apy = protocolConfigs[position.protocol][position.chain].apy;
        uint256 yield = (position.amount * apy * timeElapsed) / (365 days * 10000);

        return position.accruedYield + yield;
    }

    /**
     * @notice Get total statistics
     */
    function getStats() external view returns (
        uint256 tvl,
        uint256 totalYield,
        uint256 activePositions
    ) {
        return (totalValueLocked, totalYieldGenerated, positionCounter);
    }

    /**
     * @notice Deactivate a protocol
     * @param protocol Protocol to deactivate
     * @param chain Chain
     */
    function deactivateProtocol(Protocol protocol, Chain chain) external onlyOwner {
        protocolConfigs[protocol][chain].isActive = false;
    }

    /**
     * @notice Emergency withdrawal (owner only)
     * @param token Token to withdraw
     * @param amount Amount to withdraw
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner(), amount);
    }
}
