# BB84 Attack with Coherent Quadrature Measurement at amplitude ~1.0

### Assumptions

We will use a simplified model with the following parameters:
- Eve must obey the laws of quantum mechanics
- Eve must compute in polynomial time and memory (and so must Alice and Bob for error correction)
- We assume no side channels or imperfections in Alice and Bob’s equipment
- Alice never ~ amplitude 1.0 photons as weak coherent pulses. this attack differs than PNS though as it uses generalized measurement
- For simplicity, we assume a zero noise channel between Alice and Eve, and a zero noise channel between Eve and Bob. 
- Under normal conditions when Eve is not present, Alice and Bob expect and accept an error rate threshold. 
- Alice and Bob perform one-way 4-state BB84 with prepare & measure
- Alice and Bob do not use hardening measures for BB84 such as decoy states, and weaker (<1) coherent states of light
- Alice transmits single photon sources, with amplitude=1.0, using 4-QPSK states
- Alice and Bob use secure random number sources 
- Alice and Bob’s classical channel is authenticated. While it is fully visible to Eve, it is not malleable to Eve and she can not recover the authentication keys to manipulate the classical channel.

### Background Literature
- [Original BB84 Publication](https://arxiv.org/abs/2003.06557) 
- [Simple Proof of Security of the BB84 Quantum Key Distribution Protocol](https://arxiv.org/abs/quant-ph/0003004)
- [The Security of Practical Quantum Key Distribution](https://arxiv.org/abs/0802.4155)
- [Implementation Attacks against QKD Systems](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/Publications/Studies/QKD-Systems/QKD-Systems.pdf)
- [Quantum State Discrimination](https://arxiv.org/abs/0810.1970)

## Prepare & Measure BB84 up to check bits

<img width="567" height="539" alt="image" src="https://github.com/user-attachments/assets/0885c8b0-dba7-4ba2-ad16-86a8e7bc903e" />

## Attack Concept 

Eve performs a quadrature measurement at the helstrom limit and re-transmits the qubits to Bob, with some error. Consider that at photon amplitude 1.0 the helstrom error is approximately 0.092.  The branch diagram below shows the statistics for what Bob measures when his basis matches Alice’s 


```
Alice → Eve (quadrature measurement) → Bob

Eve's measurement:
├─ Correct (90.8%)
│   └─ Re-sends correct state → Bob receives match (90.8%)
│
└─ Incorrect (9.2%)
    └─ Re-sends wrong state
        ├─ Eve sent in Wrong basis (2/3 of errors = 6.12%)
        │   └─ Bob reads random value
        │       ├─ Correct by chance (50% = 3.06%)
        │       └─ Error detected (50% = 3.06%)
        │
        └─ Eve sent right basis, wrong value (1/3 of errors)
            └─ Bob always detects error (3.06%)

Final outcomes at Bob:
- Match alice: 90.8% + 3.06% = 93.86%
- Error alice: 3.06% + 3.06% = 6.12%
```
Next step 5 occurs and Alice reveals her Basis choices. Eve knows the following statistics about Eve’s bitstream choice when Alice’s Basis matches Bob’s basis. The knowledge is ambiguous in that Alice does not know the exact value of any of the bits, only the statistics of those values as diagramed here: 

```
           Alice reveals basis choices
                      |
                      ▼
              Eve categorizes her measurements
                   /              \
                 /                  \
    Eve's basis was correct      Eve's basis was wrong
         (93.86%)                      (6.12%)
            |                             |
      ______|______                       |
     /             \                      |
   90.8%          3.06%                   |
   Match          Error                   |
     |              |                     |
     ▼              ▼                     ▼
Eve knows:      Eve knows:           Eve knows:
Alice sent      Alice sent           • Positions of these bits
same value      opposite value       • Alice's values are 50-50
Eve measured    from Eve's            distributed 
                                     • Doesn't know specific ves
    
[Eve knows distributions but NO specific values for each]
```


A key insight emerges when considering Eve’s knowledge of Bob’s state after sifting. When Alice has announced her basis choices, Eve learns a large part of Bob’s bitstream unambiguously when the quadrature measurements were on the correct basis. Assuming 0% channel noise to Bob, the transmissions in this Basis when Alice = Eve are identical in value to what Bob has read. When Alice and Bob’s basis differs, Bob discards them, so we do not need to consider their values.  The quadrature measurement can fail on the right basis as well, but Bob will correspondingly have the same error Eve read from Alice and sent to Bob. 

In this way BB84’s step 5 upgrades the ambiguous measurement to mostly unambiguous knowledge with respect to the information shared between Eve and Bob. The amount of ambiguous information remaining has a 50% random distribution, when Alice and Bob share a basis but Eve measured the basis incorrectly. This will be approximately 2/3rd of the helstrom bound error and arguably the security of BB84 under these conditions is reduced to this quantity.


```

           Alice reveals basis choices
                      |
                      ▼
         Eve analyzes what Bob received
                   /              \
                 /                  \
    Alice & Eve basis match      Alice & Eve basis differ
         (93.86%)                      (6.12%)
            |                             |
      ______|______                       |
     /             \                      |
   90.8%          3.06%                   |
Eve sent         Eve sent                 |
correct          wrong value              |
     |              |                     |
     ▼              ▼                     ▼
Eve knows:      Eve knows:           Eve knows:
Bob received    Bob received         • Positions of these bits
EXACT value     WRONG value          • Bob's values are 50-50
Alice sent      (opposite of           distributed
                Alice's)              • Doesn't know specific values
                                       Bob received

    [Eve has PERFECT knowledge of        [Eve knows positions but
     what Bob received for ALL           NOT specific values -
     93.86% of these bits]               only 50-50 statistics]

```


The next thing to consider is the impact of quadrature measurement on the resulting QBER between Alice and Bob. We assume Eve has noiseless channels to both Alice and to Bob. 
Bob’s resulting bit matches to Alice’s transmissions will be as follows.

HBER is the helstrom bound error. 

At a rate of (1-HBER) Eve guessed the Basis and Value correctly, and Bob has the correct values.
At a rate of ⅓ HBER Eve had the wrong Value, but correct basis, and Bob has an incorrect bit value.
At a rate of ⅔ HBER Eve had the wrong Basis, and Bob has the correct bit with 50% probability.

⅓ + ⅔ * ½ = ⅔ HBER as the expected QBER between Bob and Alice. 

So if Bob and Alice set their tolerable QBER above ⅔ * HBER, Eve can intercept every qubit with quadrature measurement, and pass the check bits in step 7.

It is worth noting that Eve can also reduce her sampling rate to further reduce Alice & Bob’s QBER. If the QBER tolerance on the check bits were 5% then Eve could adjust to sampling 80% of the time (0.092*⅔ * 0.80 = 0.05 QBER) 

## Attack Results Against Cascade

**constraint solver simulation**
<img width="1483" height="992" alt="image" src="https://github.com/user-attachments/assets/acec0f48-59fb-4985-85c3-219d9f904837" />

**eve sampling rate 3d plot** 
<img width="2144" height="1788" alt="image" src="https://github.com/user-attachments/assets/3128d88f-ddc3-49fa-b659-7b91ad3c679c" />

**knowledgeknowledge_error_correction_analysis.py**
<img width="5084" height="2574" alt="image" src="https://github.com/user-attachments/assets/2ed03c56-0651-4464-8678-f45d377b7f71" />


**potential theoretical worst case bounds for secure keyrates**
<img width="2742" height="2222" alt="image" src="https://github.com/user-attachments/assets/b4ef0b35-3d1b-4f6e-a761-8a70e08584c3" />

## Using the code

```
# grab cascade
git submodule update --init --recursive 
# run the sims
PYTHONPATH=$PWD/cascade-python python3 plot_*py
```

## Attack Results Against adapative LDPC Scheme

See ldpc/, Eve is able to get 100% recovery at ~6% QBER

