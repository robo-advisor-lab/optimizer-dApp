Contracts deployed:
ChainlinkPriceOracle: 0xD73363b6b0A1cE3Fe374951b1ed5413E820850bC
MLPredictionOracle: 0xfEda0Bff2688aEDEB9830117809b2C9E33c9177C
UniswapAdapter: 0x19c144196051b72F40dF0F70a9a48b4e871cD851
SmartPortfolioManager: 0x65C7A92691ce180c32ac316bB6237EE31595408C
InvestmentVehicle: 0x61257521c8DB4C14Bd98AB713430804e0ADB18Ca
```tla
---- MODULE EnhancedFundOperations ----
EXTENDS Integers, Sequences, FiniteSets, Reals

CONSTANTS
    Investors,
    Assets,
    MaxTokens,
    MaxDuration,
    MaxRebalancingFrequency,
    MinVotingPeriod,
    MaxVotingPeriod,
    QuorumThreshold,
    ApprovalThreshold

VARIABLES
    fundManager,
    governance,
    treasury,
    mlOracle,
    uniswap,
    chainlink,
    vehicleFactory,
    investmentVehicles,
    disperseApp

vars == << fundManager, governance, treasury, mlOracle, uniswap, chainlink, vehicleFactory, investmentVehicles, disperseApp >>

\* Tipos de datos
VehicleParams == [
    duration: 1..MaxDuration,
    rebalancingFrequency: 1..MaxRebalancingFrequency,
    numTokens: 1..MaxTokens,
    tokenAddresses: SUBSET Assets
]

ProposalStatus == {"Active", "Passed", "Rejected", "Executed"}

Proposal == [
    id: Nat,
    description: STRING,
    startTime: Nat,
    endTime: Nat,
    status: ProposalStatus,
    votes: [Investors -> {"For", "Against", "Abstain"}]
]

\* Funciones auxiliares
CalculateNewWeights(prediction, currentAssets) ==
    LET totalValue == SUM({prediction[a] * currentAssets[a] : a \in Assets})
    IN [a \in Assets |-> (prediction[a] * currentAssets[a]) / totalValue]

CalculateNAV(assets, tokenAddresses) ==
    SUM({chainlink.prices[a] * assets[a] : a \in tokenAddresses})

CalculateVotingPower(investor) ==
    LET totalTokens == SUM({v.tokenHolders[investor] : v \in investmentVehicles})
    IN totalTokens / SUM({v.totalSupply : v \in investmentVehicles})

\* Acciones

InitializeFund ==
    /\ fundManager' = [
        nav: 0,
        totalSupply: 0,
        managementFee: 0.02,  \* 2% annual management fee
        performanceFee: 0.20  \* 20% performance fee
       ]
    /\ governance' = [
        proposals: {},
        votingPower: [i \in Investors |-> 0],
        nextProposalId: 1
       ]
    /\ treasury' = [
        assets: [a \in Assets |-> 0]
       ]
    /\ mlOracle' = [
        latestPrediction: [a \in Assets |-> 1]
       ]
    /\ uniswap' = [
        liquidity: [a \in Assets |-> 0],
        fees: 0.003
       ]
    /\ chainlink' = [
        prices: [a \in Assets |-> 1]
       ]
    /\ vehicleFactory' = [
        vehicleCount: 0
       ]
    /\ investmentVehicles' = {}
    /\ disperseApp' = [
        pendingDistributions: {}
       ]

CreateNewVehicle(params) ==
    /\ params \in VehicleParams
    /\ LET newVehicle == [
           id: vehicleFactory.vehicleCount + 1,
           params: params,
           nav: 0,
           totalSupply: 0,
           tokenHolders: [i \in Investors |-> 0],
           lastRebalance: 0
       ]
       IN
       /\ investmentVehicles' = investmentVehicles \union {newVehicle}
       /\ vehicleFactory' = [vehicleFactory EXCEPT !.vehicleCount = @ + 1]

DepositAssets(investor, vehicle, assetAmounts) ==
    /\ vehicle \in investmentVehicles
    /\ \A a \in DOMAIN assetAmounts : assetAmounts[a] >= 0
    /\ LET totalDeposit == SUM({chainlink.prices[a] * assetAmounts[a] : a \in DOMAIN assetAmounts})
           newTokens == IF vehicle.totalSupply = 0 THEN totalDeposit
                        ELSE totalDeposit * vehicle.totalSupply / vehicle.nav
       IN
       /\ treasury' = [treasury EXCEPT !.assets = [a \in Assets |-> 
                       treasury.assets[a] + IF a \in DOMAIN assetAmounts THEN assetAmounts[a] ELSE 0]]
       /\ investmentVehicles' = (investmentVehicles \ {vehicle}) \union 
                                {[vehicle EXCEPT 
                                    !.nav = @ + totalDeposit,
                                    !.totalSupply = @ + newTokens,
                                    !.tokenHolders = [i \in Investors |-> 
                                        IF i = investor THEN @[i] + newTokens ELSE @[i]]
                                ]}
       /\ fundManager' = [fundManager EXCEPT !.nav = @ + totalDeposit]

WithdrawAssets(investor, vehicle, tokenAmount) ==
    /\ vehicle \in investmentVehicles
    /\ tokenAmount <= vehicle.tokenHolders[investor]
    /\ LET withdrawalShare == tokenAmount / vehicle.totalSupply
           withdrawalAmount == [a \in Assets |-> withdrawalShare * treasury.assets[a]]
           totalWithdrawal == SUM({chainlink.prices[a] * withdrawalAmount[a] : a \in Assets})
       IN
       /\ treasury' = [treasury EXCEPT !.assets = [a \in Assets |-> @[a] - withdrawalAmount[a]]]
       /\ investmentVehicles' = (investmentVehicles \ {vehicle}) \union 
                                {[vehicle EXCEPT 
                                    !.nav = @ - totalWithdrawal,
                                    !.totalSupply = @ - tokenAmount,
                                    !.tokenHolders = [i \in Investors |-> 
                                        IF i = investor THEN @[i] - tokenAmount ELSE @[i]]
                                ]}
       /\ fundManager' = [fundManager EXCEPT !.nav = @ - totalWithdrawal]

UpdateChainlinkPrices ==
    \E newPrices \in [Assets -> Real] :
        chainlink' = [chainlink EXCEPT !.prices = newPrices]

GetMLPrediction ==
    \E prediction \in [Assets -> Real] :
        mlOracle' = [mlOracle EXCEPT !.latestPrediction = prediction]

Rebalance(vehicle) ==
    /\ vehicle \in investmentVehicles
    /\ vehicle.lastRebalance + vehicle.params.rebalancingFrequency <= Now
    /\ LET prediction == mlOracle.latestPrediction
           currentAssets == [a \in Assets |-> treasury.assets[a] * (vehicle.nav / fundManager.nav)]
           newWeights == CalculateNewWeights(prediction, currentAssets)
           tradesToExecute == [a \in Assets |-> newWeights[a] - currentAssets[a]]
           tradingFees == SUM({ABS(tradesToExecute[a]) * uniswap.fees : a \in Assets})
       IN
       /\ treasury' = [treasury EXCEPT !.assets = [a \in Assets |-> @[a] + tradesToExecute[a]]]
       /\ uniswap' = [uniswap EXCEPT !.liquidity = [a \in Assets |-> @[a] - tradesToExecute[a]]]
       /\ investmentVehicles' = (investmentVehicles \ {vehicle}) \union 
                                {[vehicle EXCEPT 
                                    !.nav = @ - tradingFees,
                                    !.lastRebalance = Now
                                ]}
       /\ fundManager' = [fundManager EXCEPT !.nav = @ - tradingFees]

UpdateNAV(vehicle) ==
    /\ vehicle \in investmentVehicles
    /\ LET newNAV == CalculateNAV(treasury.assets, vehicle.params.tokenAddresses)
           managementFee == (newNAV - vehicle.nav) * fundManager.managementFee * (Now - vehicle.lastRebalance) / 365
           performanceFee == MAX(0, (newNAV - vehicle.nav - managementFee) * fundManager.performanceFee)
           totalFees == managementFee + performanceFee
       IN
       /\ investmentVehicles' = (investmentVehicles \ {vehicle}) \union 
                                {[vehicle EXCEPT 
                                    !.nav = newNAV - totalFees,
                                    !.lastRebalance = Now
                                ]}
       /\ fundManager' = [fundManager EXCEPT !.nav = @ - totalFees]

CreateGovernanceProposal(description, votingPeriod) ==
    /\ votingPeriod \in MinVotingPeriod..MaxVotingPeriod
    /\ LET newProposal == [
           id: governance.nextProposalId,
           description: description,
           startTime: Now,
           endTime: Now + votingPeriod,
           status: "Active",
           votes: [i \in Investors |-> "Abstain"]
       ]
       IN
       /\ governance' = [governance EXCEPT 
           !.proposals = @ \union {newProposal},
           !.nextProposalId = @ + 1
       ]

Vote(investor, proposalId, voteChoice) ==
    /\ \E proposal \in governance.proposals : 
        /\ proposal.id = proposalId
        /\ proposal.status = "Active"
        /\ Now <= proposal.endTime
        /\ LET updatedProposal == [proposal EXCEPT !.votes = [@ EXCEPT ![investor] = voteChoice]]
           IN
           governance' = [governance EXCEPT !.proposals = (@ \ {proposal}) \union {updatedProposal}]

ExecuteProposal(proposalId) ==
    /\ \E proposal \in governance.proposals :
        /\ proposal.id = proposalId
        /\ proposal.status = "Active"
        /\ Now > proposal.endTime
        /\ LET totalVotes == SUM({CalculateVotingPower(i) : i \in Investors})
               forVotes == SUM({CalculateVotingPower(i) : i \in Investors WHERE proposal.votes[i] = "For"})
               againstVotes == SUM({CalculateVotingPower(i) : i \in Investors WHERE proposal.votes[i] = "Against"})
           IN
           /\ totalVotes >= QuorumThreshold
           /\ forVotes / (forVotes + againstVotes) >= ApprovalThreshold
        /\ LET updatedProposal == [proposal EXCEPT !.status = "Passed"]
           IN
           /\ governance' = [governance EXCEPT !.proposals = (@ \ {proposal}) \union {updatedProposal}]
           \* Here you would implement the actual execution of the proposal
           /\ UNCHANGED << fundManager, treasury, mlOracle, uniswap, chainlink, vehicleFactory, investmentVehicles, disperseApp >>

LiquidateVehicle(vehicle) ==
    /\ vehicle \in investmentVehicles
    /\ vehicle.params.duration <= Now
    /\ LET liquidationAmount == [a \in Assets |-> treasury.assets[a] * (vehicle.nav / fundManager.nav)]
           distribution == [i \in Investors |-> 
               [a \in Assets |-> liquidationAmount[a] * (vehicle.tokenHolders[i] / vehicle.totalSupply)]]
       IN
       /\ treasury' = [treasury EXCEPT !.assets = [a \in Assets |-> @[a] - liquidationAmount[a]]]
       /\ investmentVehicles' = investmentVehicles \ {vehicle}
       /\ disperseApp' = [disperseApp EXCEPT !.pendingDistributions = @ \union {distribution}]
       /\ fundManager' = [fundManager EXCEPT !.nav = @ - vehicle.nav]

DistributeAssets ==
    /\ disperseApp.pendingDistributions /= {}
    /\ LET distribution == CHOOSE d \in disperseApp.pendingDistributions : TRUE
       IN
       /\ disperseApp' = [disperseApp EXCEPT !.pendingDistributions = @ \ {distribution}]
       \* Here, you would implement the actual distribution of assets to investors
       /\ UNCHANGED << fundManager, governance, treasury, mlOracle, uniswap, chainlink, vehicleFactory, investmentVehicles >>

Next ==
    \/ (\E params \in VehicleParams : CreateNewVehicle(params))
    \/ (\E i \in Investors, v \in investmentVehicles, amounts \in [Assets -> Nat] : DepositAssets(i, v, amounts))
    \/ (\E i \in Investors, v \in investmentVehicles, amount \in Nat : WithdrawAssets(i, v, amount))
    \/ UpdateChainlinkPrices
    \/ GetMLPrediction
    \/ (\E v \in investmentVehicles : Rebalance(v))
    \/ (\E v \in investmentVehicles : UpdateNAV(v))
    \/ (\E desc \in STRING, period \in MinVotingPeriod..MaxVotingPeriod : CreateGovernanceProposal(desc, period))
    \/ (\E i \in Investors, pid \in Nat, choice \in {"For", "Against", "Abstain"} : Vote(i, pid, choice))
    \/ (\E pid \in Nat : ExecuteProposal(pid))
    \/ (\E v \in investmentVehicles : LiquidateVehicle(v))
    \/ DistributeAssets

Spec == Init /\ [][Next]_vars

====
```
