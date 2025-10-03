"""
Blockchain Interface for AI DAO Hedge Fund
Handles Web3 interactions with smart contracts
"""

from web3 import Web3
from eth_account import Account
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BlockchainInterface:
    """Interface to interact with DAO smart contracts"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize blockchain connection"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['blockchain']
        self.w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))

        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to blockchain network")

        logger.info(f"Connected to {self.config['network']} network")

        # Load contract ABIs and addresses
        self.contracts = {}
        self._load_contracts()

    def _load_contracts(self):
        """Load deployed contract addresses and ABIs"""
        # In production, load from deployment artifacts
        # For now, we'll set placeholders
        self.contracts = {
            'governance': {
                'address': self.config.get('dao_address'),
                'abi': None  # Load from compiled artifacts
            },
            'treasury': {
                'address': self.config.get('treasury_address'),
                'abi': None
            }
        }

    def create_proposal(
        self,
        private_key: str,
        description: str,
        proposal_type: int,
        data: bytes
    ) -> str:
        """
        Create a governance proposal

        Args:
            private_key: Proposer's private key
            description: Human-readable proposal description
            proposal_type: Type of proposal (0-5)
            data: Encoded proposal data

        Returns:
            Transaction hash
        """
        account = Account.from_key(private_key)

        # Build transaction
        governance = self.w3.eth.contract(
            address=self.contracts['governance']['address'],
            abi=self.contracts['governance']['abi']
        )

        txn = governance.functions.createProposal(
            description,
            proposal_type,
            data
        ).build_transaction({
            'from': account.address,
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price
        })

        # Sign and send
        signed_txn = self.w3.eth.account.sign_transaction(txn, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        logger.info(f"Proposal created: {tx_hash.hex()}")
        return tx_hash.hex()

    def cast_vote(
        self,
        private_key: str,
        proposal_id: int,
        support: bool
    ) -> str:
        """
        Vote on a proposal

        Args:
            private_key: Voter's private key
            proposal_id: ID of proposal to vote on
            support: True for yes, False for no

        Returns:
            Transaction hash
        """
        account = Account.from_key(private_key)

        governance = self.w3.eth.contract(
            address=self.contracts['governance']['address'],
            abi=self.contracts['governance']['abi']
        )

        txn = governance.functions.castVote(
            proposal_id,
            support
        ).build_transaction({
            'from': account.address,
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })

        signed_txn = self.w3.eth.account.sign_transaction(txn, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        logger.info(f"Vote cast on proposal {proposal_id}: {tx_hash.hex()}")
        return tx_hash.hex()

    def execute_proposal(self, private_key: str, proposal_id: int) -> str:
        """Execute a passed proposal"""
        account = Account.from_key(private_key)

        governance = self.w3.eth.contract(
            address=self.contracts['governance']['address'],
            abi=self.contracts['governance']['abi']
        )

        txn = governance.functions.executeProposal(
            proposal_id
        ).build_transaction({
            'from': account.address,
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price
        })

        signed_txn = self.w3.eth.account.sign_transaction(txn, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        logger.info(f"Proposal {proposal_id} executed: {tx_hash.hex()}")
        return tx_hash.hex()

    def get_proposal(self, proposal_id: int) -> Dict:
        """Get proposal details"""
        governance = self.w3.eth.contract(
            address=self.contracts['governance']['address'],
            abi=self.contracts['governance']['abi']
        )

        proposal = governance.functions.getProposal(proposal_id).call()

        return {
            'proposer': proposal[0],
            'description': proposal[1],
            'proposal_type': proposal[2],
            'start_time': proposal[3],
            'end_time': proposal[4],
            'for_votes': proposal[5],
            'against_votes': proposal[6],
            'executed': proposal[7],
            'canceled': proposal[8]
        }

    def record_trade(
        self,
        private_key: str,
        agent_id: int,
        pnl: int
    ) -> str:
        """
        Record a trade on-chain

        Args:
            private_key: Admin private key
            agent_id: ID of the agent
            pnl: Profit/Loss in basis points

        Returns:
            Transaction hash
        """
        account = Account.from_key(private_key)

        treasury = self.w3.eth.contract(
            address=self.contracts['treasury']['address'],
            abi=self.contracts['treasury']['abi']
        )

        txn = treasury.functions.recordTrade(
            agent_id,
            pnl
        ).build_transaction({
            'from': account.address,
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'gas': 150000,
            'gasPrice': self.w3.eth.gas_price
        })

        signed_txn = self.w3.eth.account.sign_transaction(txn, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        logger.info(f"Trade recorded for agent {agent_id}: PnL={pnl}, tx={tx_hash.hex()}")
        return tx_hash.hex()

    def get_agent_performance(self, agent_id: int) -> Dict:
        """Get agent performance metrics from blockchain"""
        treasury = self.w3.eth.contract(
            address=self.contracts['treasury']['address'],
            abi=self.contracts['treasury']['abi']
        )

        performance = treasury.functions.getAgentPerformance(agent_id).call()

        return {
            'name': performance[0],
            'is_active': performance[1],
            'allocation': performance[2],
            'total_trades': performance[3],
            'total_pnl': performance[4],
            'avg_pnl_per_trade': performance[5]
        }

    def deposit(self, private_key: str, amount_eth: float) -> str:
        """Deposit funds to treasury"""
        account = Account.from_key(private_key)
        amount_wei = self.w3.to_wei(amount_eth, 'ether')

        treasury = self.w3.eth.contract(
            address=self.contracts['treasury']['address'],
            abi=self.contracts['treasury']['abi']
        )

        txn = treasury.functions.deposit().build_transaction({
            'from': account.address,
            'value': amount_wei,
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })

        signed_txn = self.w3.eth.account.sign_transaction(txn, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        logger.info(f"Deposited {amount_eth} ETH: {tx_hash.hex()}")
        return tx_hash.hex()

    def withdraw(self, private_key: str, shares: int) -> str:
        """Withdraw shares from treasury"""
        account = Account.from_key(private_key)

        treasury = self.w3.eth.contract(
            address=self.contracts['treasury']['address'],
            abi=self.contracts['treasury']['abi']
        )

        txn = treasury.functions.withdraw(shares).build_transaction({
            'from': account.address,
            'nonce': self.w3.eth.get_transaction_count(account.address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })

        signed_txn = self.w3.eth.account.sign_transaction(txn, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        logger.info(f"Withdrew {shares} shares: {tx_hash.hex()}")
        return tx_hash.hex()

    def get_investor_position(self, address: str) -> Dict:
        """Get investor's position"""
        treasury = self.w3.eth.contract(
            address=self.contracts['treasury']['address'],
            abi=self.contracts['treasury']['abi']
        )

        position = treasury.functions.getInvestorPosition(address).call()

        return {
            'shares': position[0],
            'current_value': position[1],
            'deposited': position[2],
            'unrealized_pnl': position[3]
        }

    def listen_for_events(self, event_name: str, callback):
        """
        Listen for smart contract events

        Args:
            event_name: Name of the event to listen for
            callback: Function to call when event is detected
        """
        # Implement event filtering and callback
        governance = self.w3.eth.contract(
            address=self.contracts['governance']['address'],
            abi=self.contracts['governance']['abi']
        )

        event_filter = governance.events[event_name].create_filter(fromBlock='latest')

        while True:
            for event in event_filter.get_new_entries():
                callback(event)


class MockBlockchainInterface:
    """Mock interface for testing without actual blockchain"""

    def __init__(self):
        self.proposals = {}
        self.trades = []
        self.balances = {}
        logger.info("Using mock blockchain interface for testing")

    def create_proposal(self, private_key: str, description: str,
                       proposal_type: int, data: bytes) -> str:
        proposal_id = len(self.proposals)
        self.proposals[proposal_id] = {
            'description': description,
            'type': proposal_type,
            'votes_for': 0,
            'votes_against': 0
        }
        return f"mock_tx_hash_{proposal_id}"

    def cast_vote(self, private_key: str, proposal_id: int, support: bool) -> str:
        if support:
            self.proposals[proposal_id]['votes_for'] += 1
        else:
            self.proposals[proposal_id]['votes_against'] += 1
        return f"mock_vote_tx_{proposal_id}"

    def record_trade(self, private_key: str, agent_id: int, pnl: int) -> str:
        self.trades.append({'agent_id': agent_id, 'pnl': pnl})
        return f"mock_trade_tx_{len(self.trades)}"

    def get_agent_performance(self, agent_id: int) -> Dict:
        agent_trades = [t for t in self.trades if t['agent_id'] == agent_id]
        total_pnl = sum(t['pnl'] for t in agent_trades)

        return {
            'name': f'Agent_{agent_id}',
            'is_active': True,
            'allocation': 3333,
            'total_trades': len(agent_trades),
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl // len(agent_trades) if agent_trades else 0
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Use mock for development
    blockchain = MockBlockchainInterface()

    # Create a proposal
    tx_hash = blockchain.create_proposal(
        "mock_private_key",
        "Enable new arbitrage agent",
        0,  # ENABLE_AGENT
        b""
    )
    print(f"Proposal created: {tx_hash}")

    # Record a trade
    tx_hash = blockchain.record_trade("mock_private_key", 0, 250)  # 2.5% profit
    print(f"Trade recorded: {tx_hash}")

    # Get performance
    perf = blockchain.get_agent_performance(0)
    print(f"Agent performance: {perf}")
