// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title MultiChainBridge
 * @notice Cross-chain bridge for AI DAO fund transfers across Ethereum, Polygon, Arbitrum
 * @dev Uses a lock-and-mint mechanism with validator attestations
 */
contract MultiChainBridge is ReentrancyGuard, Ownable, Pausable {

    // Supported chains
    enum Chain {
        ETHEREUM,
        POLYGON,
        ARBITRUM,
        BSC,
        AVALANCHE
    }

    struct BridgeTransaction {
        address sender;
        address recipient;
        uint256 amount;
        Chain sourceChain;
        Chain destinationChain;
        uint256 timestamp;
        bool isCompleted;
        uint256 nonce;
    }

    struct ValidatorAttestation {
        address validator;
        bytes32 txHash;
        bool approved;
        uint256 timestamp;
    }

    // State variables
    mapping(bytes32 => BridgeTransaction) public bridgeTransactions;
    mapping(bytes32 => mapping(address => bool)) public validatorAttestations;
    mapping(bytes32 => uint256) public attestationCount;
    mapping(address => bool) public validators;
    mapping(Chain => address) public chainContracts;

    uint256 public validatorCount;
    uint256 public requiredAttestations;
    uint256 public bridgeFeePercent = 10; // 0.1% fee (in basis points)
    uint256 public totalBridged;
    uint256 public transactionNonce;

    IERC20 public token;

    // Events
    event BridgeInitiated(
        bytes32 indexed txHash,
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        Chain sourceChain,
        Chain destinationChain,
        uint256 nonce
    );

    event BridgeCompleted(
        bytes32 indexed txHash,
        address indexed recipient,
        uint256 amount
    );

    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);
    event AttestationSubmitted(bytes32 indexed txHash, address indexed validator);
    event ChainContractUpdated(Chain indexed chain, address contractAddress);

    constructor(
        address _token,
        uint256 _requiredAttestations
    ) Ownable(msg.sender) {
        token = IERC20(_token);
        requiredAttestations = _requiredAttestations;
    }

    /**
     * @notice Initiate a cross-chain bridge transfer
     * @param recipient Address to receive funds on destination chain
     * @param amount Amount to bridge
     * @param destinationChain Target chain
     */
    function initiateBridge(
        address recipient,
        uint256 amount,
        Chain destinationChain
    ) external nonReentrant whenNotPaused returns (bytes32) {
        require(amount > 0, "Amount must be greater than 0");
        require(recipient != address(0), "Invalid recipient");

        // Calculate fee
        uint256 fee = (amount * bridgeFeePercent) / 10000;
        uint256 netAmount = amount - fee;

        // Lock tokens on source chain
        require(
            token.transferFrom(msg.sender, address(this), amount),
            "Token transfer failed"
        );

        // Generate unique transaction hash
        bytes32 txHash = keccak256(
            abi.encodePacked(
                msg.sender,
                recipient,
                amount,
                destinationChain,
                block.timestamp,
                transactionNonce
            )
        );

        // Create bridge transaction
        bridgeTransactions[txHash] = BridgeTransaction({
            sender: msg.sender,
            recipient: recipient,
            amount: netAmount,
            sourceChain: Chain.ETHEREUM, // Adjust based on deployment
            destinationChain: destinationChain,
            timestamp: block.timestamp,
            isCompleted: false,
            nonce: transactionNonce
        });

        totalBridged += netAmount;
        transactionNonce++;

        emit BridgeInitiated(
            txHash,
            msg.sender,
            recipient,
            netAmount,
            Chain.ETHEREUM,
            destinationChain,
            transactionNonce - 1
        );

        return txHash;
    }

    /**
     * @notice Validator attests to a bridge transaction
     * @param txHash Transaction hash to attest
     */
    function submitAttestation(bytes32 txHash) external {
        require(validators[msg.sender], "Not a validator");
        require(!bridgeTransactions[txHash].isCompleted, "Transaction already completed");
        require(!validatorAttestations[txHash][msg.sender], "Already attested");

        validatorAttestations[txHash][msg.sender] = true;
        attestationCount[txHash]++;

        emit AttestationSubmitted(txHash, msg.sender);

        // Auto-complete if threshold reached
        if (attestationCount[txHash] >= requiredAttestations) {
            _completeBridge(txHash);
        }
    }

    /**
     * @notice Complete bridge transaction after sufficient attestations
     * @param txHash Transaction hash to complete
     */
    function _completeBridge(bytes32 txHash) internal {
        BridgeTransaction storage bridgeTx = bridgeTransactions[txHash];

        require(!bridgeTx.isCompleted, "Already completed");
        require(attestationCount[txHash] >= requiredAttestations, "Insufficient attestations");

        bridgeTx.isCompleted = true;

        // On destination chain, this would mint tokens
        // On source chain (this example), we release locked tokens
        require(
            token.transfer(bridgeTx.recipient, bridgeTx.amount),
            "Token transfer failed"
        );

        emit BridgeCompleted(txHash, bridgeTx.recipient, bridgeTx.amount);
    }

    /**
     * @notice Add a validator
     * @param validator Address of new validator
     */
    function addValidator(address validator) external onlyOwner {
        require(validator != address(0), "Invalid validator address");
        require(!validators[validator], "Already a validator");

        validators[validator] = true;
        validatorCount++;

        emit ValidatorAdded(validator);
    }

    /**
     * @notice Remove a validator
     * @param validator Address of validator to remove
     */
    function removeValidator(address validator) external onlyOwner {
        require(validators[validator], "Not a validator");

        validators[validator] = false;
        validatorCount--;

        emit ValidatorRemoved(validator);
    }

    /**
     * @notice Update contract address for a specific chain
     * @param chain Chain to update
     * @param contractAddress New contract address
     */
    function updateChainContract(Chain chain, address contractAddress) external onlyOwner {
        require(contractAddress != address(0), "Invalid contract address");
        chainContracts[chain] = contractAddress;

        emit ChainContractUpdated(chain, contractAddress);
    }

    /**
     * @notice Update bridge fee
     * @param newFeePercent New fee in basis points (100 = 1%)
     */
    function updateBridgeFee(uint256 newFeePercent) external onlyOwner {
        require(newFeePercent <= 500, "Fee too high"); // Max 5%
        bridgeFeePercent = newFeePercent;
    }

    /**
     * @notice Update required attestations
     * @param newRequired New required attestation count
     */
    function updateRequiredAttestations(uint256 newRequired) external onlyOwner {
        require(newRequired > 0, "Must require at least 1 attestation");
        require(newRequired <= validatorCount, "Cannot require more than validator count");
        requiredAttestations = newRequired;
    }

    /**
     * @notice Emergency pause
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @notice Unpause
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @notice Get bridge transaction details
     * @param txHash Transaction hash
     */
    function getBridgeTransaction(bytes32 txHash)
        external
        view
        returns (BridgeTransaction memory)
    {
        return bridgeTransactions[txHash];
    }

    /**
     * @notice Check if validator has attested to a transaction
     * @param txHash Transaction hash
     * @param validator Validator address
     */
    function hasAttested(bytes32 txHash, address validator) external view returns (bool) {
        return validatorAttestations[txHash][validator];
    }

    /**
     * @notice Get attestation count for a transaction
     * @param txHash Transaction hash
     */
    function getAttestationCount(bytes32 txHash) external view returns (uint256) {
        return attestationCount[txHash];
    }

    /**
     * @notice Emergency withdrawal (owner only)
     * @param amount Amount to withdraw
     */
    function emergencyWithdraw(uint256 amount) external onlyOwner {
        require(token.transfer(owner(), amount), "Transfer failed");
    }
}
