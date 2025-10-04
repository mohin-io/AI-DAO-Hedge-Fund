"""
DAO Governance - Interactive blockchain governance interface
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def render():
    """Render the DAO governance page"""

    st.title("⛓️ DAO Governance")
    st.markdown("Decentralized governance for AI trading strategies and fund management")

    # Connection Status
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("DAO Members", "142", delta="+5 this week")

    with col2:
        st.metric("Active Proposals", "3", delta="2 pending")

    with col3:
        st.metric("Total Votes Cast", "1,284", delta="+47 today")

    with col4:
        st.metric("Treasury Balance", "$1.25M", delta="+3.5%")

    st.markdown("---")

    # Blockchain Connection
    st.markdown("### 🔗 Blockchain Connection")

    col1, col2 = st.columns([2, 1])

    with col1:
        network = st.selectbox(
            "Select Network",
            ["Ethereum Mainnet", "Sepolia Testnet", "Polygon", "Arbitrum"],
            index=1
        )

        wallet_address = st.text_input(
            "Wallet Address",
            value="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            disabled=True
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        connection_status = st.empty()
        connection_status.success("✅ Connected to Sepolia Testnet")

        if st.button("🔌 Disconnect", use_container_width=True):
            connection_status.warning("⚠️ Disconnected")

    # Smart Contract Addresses
    with st.expander("📜 Smart Contract Addresses"):
        col1, col2 = st.columns(2)

        with col1:
            st.code("DAOGovernance: 0x5FbDB2315678afecb367f032d93F642f64180aa3", language="text")
            st.code("TreasuryManager: 0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512", language="text")

        with col2:
            st.code("AgentRegistry: 0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0", language="text")
            st.markdown("[View on Etherscan](https://sepolia.etherscan.io) →")

    st.markdown("---")

    # Active Proposals
    st.markdown("### 📋 Active Proposals")

    proposals_data = [
        {
            "id": 13,
            "title": "Add DeFi Yield Farming Strategy",
            "description": "Integrate yield farming capabilities to generate passive income from idle assets in the treasury.",
            "proposer": "0x742d...0bEb",
            "created": datetime.now() - timedelta(days=1),
            "deadline": datetime.now() + timedelta(days=2),
            "status": "🗳️ VOTING",
            "votes_for": 89,
            "votes_against": 23,
            "quorum": 142 * 0.10,  # 10% quorum
            "type": "Strategy Addition"
        },
        {
            "id": 12,
            "title": "Increase Hedging Agent Allocation to 40%",
            "description": "Due to increased market volatility, propose to increase hedging agent allocation from 30% to 40%.",
            "proposer": "0x1a2b...3c4d",
            "created": datetime.now() - timedelta(days=4),
            "deadline": datetime.now() - timedelta(hours=2),
            "status": "✅ PASSED",
            "votes_for": 112,
            "votes_against": 31,
            "quorum": 142 * 0.10,
            "type": "Allocation Adjustment"
        },
        {
            "id": 11,
            "title": "Emergency Stop - Market Circuit Breaker",
            "description": "Activate emergency stop for all trading agents due to extreme market conditions.",
            "proposer": "0x5e6f...7g8h",
            "created": datetime.now() - timedelta(days=7),
            "deadline": datetime.now() - timedelta(days=4),
            "status": "❌ REJECTED",
            "votes_for": 45,
            "votes_against": 97,
            "quorum": 142 * 0.10,
            "type": "Emergency Action"
        }
    ]

    for proposal in proposals_data:
        with st.expander(f"**Proposal #{proposal['id']}: {proposal['title']}** - {proposal['status']}", expanded=(proposal['status'] == "🗳️ VOTING")):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Description:**")
                st.write(proposal['description'])

                st.markdown(f"**Type:** `{proposal['type']}`")
                st.markdown(f"**Proposer:** `{proposal['proposer']}`")
                st.markdown(f"**Created:** {proposal['created'].strftime('%Y-%m-%d %H:%M')}")
                st.markdown(f"**Deadline:** {proposal['deadline'].strftime('%Y-%m-%d %H:%M')}")

                # Voting progress
                total_votes = proposal['votes_for'] + proposal['votes_against']
                approval_rate = proposal['votes_for'] / total_votes if total_votes > 0 else 0
                quorum_met = total_votes >= proposal['quorum']

                st.markdown(f"**Votes:** {proposal['votes_for']} FOR / {proposal['votes_against']} AGAINST")
                st.markdown(f"**Approval Rate:** {approval_rate*100:.1f}%")
                st.markdown(f"**Quorum:** {'✅ Met' if quorum_met else '❌ Not Met'} ({total_votes}/{proposal['quorum']:.0f})")

                # Progress bar
                st.progress(approval_rate, text=f"{approval_rate*100:.1f}% Approval")

            with col2:
                if proposal['status'] == "🗳️ VOTING":
                    st.markdown("**Cast Your Vote:**")

                    vote_choice = st.radio(
                        f"Vote on Proposal #{proposal['id']}",
                        ["👍 For", "👎 Against"],
                        key=f"vote_{proposal['id']}"
                    )

                    voting_power = st.number_input(
                        "Voting Power",
                        min_value=1,
                        max_value=100,
                        value=10,
                        key=f"power_{proposal['id']}"
                    )

                    if st.button(f"🗳️ Submit Vote", key=f"submit_{proposal['id']}", use_container_width=True):
                        st.success(f"Vote cast: {vote_choice}")

                elif proposal['status'] == "✅ PASSED":
                    st.success("Proposal Passed")
                    if st.button("⚡ Execute", key=f"exec_{proposal['id']}", use_container_width=True):
                        st.info("Executing proposal on-chain...")

                else:
                    st.error("Proposal Rejected")

    st.markdown("---")

    # Create New Proposal
    st.markdown("### ✍️ Create New Proposal")

    with st.form("new_proposal"):
        prop_title = st.text_input("Proposal Title")

        prop_description = st.text_area(
            "Description",
            help="Provide detailed description of the proposal"
        )

        prop_type = st.selectbox(
            "Proposal Type",
            [
                "Enable/Disable Agent",
                "Allocation Adjustment",
                "Strategy Addition",
                "Fee Adjustment",
                "Emergency Action",
                "Other"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:
            voting_period = st.selectbox(
                "Voting Period",
                ["1 day", "3 days", "7 days", "14 days"],
                index=1
            )

        with col2:
            required_quorum = st.slider(
                "Required Quorum (%)",
                min_value=5,
                max_value=50,
                value=10,
                step=5
            )

        submit_proposal = st.form_submit_button("📤 Submit Proposal")

        if submit_proposal:
            if prop_title and prop_description:
                st.success(f"✅ Proposal '{prop_title}' submitted successfully!")
                st.info("Transaction hash: 0xabc123...")
            else:
                st.error("Please fill in all required fields")

    st.markdown("---")

    # Voting History
    st.markdown("### 📊 Voting Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Proposal outcomes
        outcomes_df = pd.DataFrame({
            'Outcome': ['Passed', 'Rejected', 'Pending'],
            'Count': [8, 3, 3]
        })

        fig_outcomes = go.Figure(data=[go.Pie(
            labels=outcomes_df['Outcome'],
            values=outcomes_df['Count'],
            marker_colors=['#00ff00', '#ff0000', '#ffff00'],
            hole=0.4
        )])

        fig_outcomes.update_layout(
            title='Proposal Outcomes',
            height=350,
            annotations=[dict(text=f'{outcomes_df["Count"].sum()}<br>Total', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )

        st.plotly_chart(fig_outcomes, use_container_width=True)

    with col2:
        # Proposal types
        types_df = pd.DataFrame({
            'Type': ['Allocation', 'Strategy', 'Emergency', 'Fee', 'Other'],
            'Count': [5, 4, 2, 2, 1]
        })

        fig_types = go.Figure(data=[
            go.Bar(
                x=types_df['Type'],
                y=types_df['Count'],
                marker_color='#667eea',
                text=types_df['Count'],
                textposition='outside'
            )
        ])

        fig_types.update_layout(
            title='Proposal Types',
            height=350,
            yaxis_title='Count',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )

        st.plotly_chart(fig_types, use_container_width=True)

    # Voting participation over time
    st.markdown("### 📈 Voting Participation Trend")

    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    participation = 80 + 15 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 3, len(dates))
    participation = np.clip(participation, 0, 100)

    fig_participation = go.Figure()

    fig_participation.add_trace(go.Scatter(
        x=dates,
        y=participation,
        mode='lines+markers',
        name='Participation Rate',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))

    # Add average line
    avg_participation = participation.mean()
    fig_participation.add_hline(
        y=avg_participation,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: {avg_participation:.1f}%",
        annotation_position="right"
    )

    fig_participation.update_layout(
        height=350,
        yaxis=dict(title='Participation Rate (%)', showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    st.plotly_chart(fig_participation, use_container_width=True)

    # DAO Member Leaderboard
    st.markdown("### 🏆 Top DAO Contributors")

    leaderboard_df = pd.DataFrame({
        'Rank': [1, 2, 3, 4, 5],
        'Address': [
            '0x742d...0bEb',
            '0x1a2b...3c4d',
            '0x5e6f...7g8h',
            '0x9i0j...1k2l',
            '0x3m4n...5o6p'
        ],
        'Proposals Created': [12, 8, 7, 5, 4],
        'Votes Cast': [156, 142, 128, 98, 87],
        'Voting Power': [250, 180, 150, 120, 100],
        'Reputation': [98, 92, 88, 85, 82]
    })

    st.dataframe(
        leaderboard_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Reputation': st.column_config.ProgressColumn(
                'Reputation',
                format='%d',
                min_value=0,
                max_value=100,
            )
        }
    )

    # Treasury Information
    st.markdown("---")
    st.markdown("### 💰 Treasury Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Treasury", "$1,247,893", delta="+$42,156")
        st.metric("Management Fee", "2% annually", delta="Standard")

    with col2:
        st.metric("Performance Fee", "20%", delta="On profits")
        st.metric("Fee Collection", "$24,958", delta="This month")

    with col3:
        st.metric("Pending Withdrawals", "$12,500", delta="2 requests")
        st.metric("Reserve Ratio", "15%", delta="Healthy")

    st.markdown("---")

    # Governance Parameters
    st.markdown("### ⚙️ Governance Parameters")

    with st.expander("View Current Parameters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **Voting Parameters**
            - Voting Period: 3 days
            - Quorum Requirement: 10%
            - Approval Threshold: 50%
            - Proposal Threshold: 1000 tokens
            """)

        with col2:
            st.markdown("""
            **Economic Parameters**
            - Management Fee: 2% annual
            - Performance Fee: 20%
            - Withdrawal Lock: 7 days
            - Min Stake: 0.1 ETH
            """)

        with col3:
            st.markdown("""
            **Agent Parameters**
            - Max Agent Count: 10
            - Min Agent Stake: 1 ETH
            - Performance Window: 30 days
            - Reputation Decay: 1% monthly
            """)
