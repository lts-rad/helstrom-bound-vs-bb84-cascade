# BB84 Attack with Coherent Quadrature Measurement at amplitude ~1.0

This repository accompanies the work presented at SPIE Photonics (https://spie.org/defense-security/presentation/A-generalized-measurement-key-intercept-attack-at-6-QBER-against/14021-18).

## Attack Concept

This differs from the textbook Intercept & Resend attack in that Eve does not 'assume' a basis, and instead performs a POVM measure.  

The attack relies on a weakened protocol that has no decoys, mean photon numbers between 0.6-1.0, and does not employ phase randomization.

In particular the POVM needs coherences in photon number to achieve quite a low minimum discrimination error, which phase randomization disables. This is known insecure and was analyzed by Lo & Preskill (https://arxiv.org/pdf/quant-ph/0504209) where no security is attainable above 0.5 for QPSK, regardless of the reconciliation efficiency. See figure below

<img width="685" height="352" alt="image" src="https://github.com/user-attachments/assets/6c3d886b-e004-4a9f-9a44-c1187d46e858" />


### POVM Attack Scaling Overview

<img width="761" height="440" alt="image" src="https://github.com/user-attachments/assets/b512f2f6-75bd-4dec-85eb-65261c62804d" />

Notice that the attack achieves a measured QBER notably lower than the Helstrom bound minimum error, even though Eve resends every packet.

<img width="644" height="363" alt="image" src="https://github.com/user-attachments/assets/8038132d-66f8-4dbc-9f7b-d012cd12ca0f" />

In the figure above it's easy to see that polarization encoding  (dashed lines) has more basis overlap than phase encoding (solid lines).
This is intuitive as the angular separation between the encodings varies from 45 for polarization to 90 degrees for typical QPSK phase encoding.

Another aspect to understand is that cross-basis overlap reduces as mean photon number increases, making the states easier to discriminate with lower minimum error.
The attack operates somewhere between the yellow and red dashed lines above. 

<img width="389" height="192" alt="image" src="https://github.com/user-attachments/assets/71b17c12-ea28-4ea2-8524-2f837ea1a100" />

The cross basis overlap does vary based on mean photon number as seen in the table above

<img width="1019" height="657" alt="image" src="https://github.com/user-attachments/assets/cd51d343-122c-401a-ba8b-cfbfe997fe9a" />


And the above 3D Plot shows how the attack operates at various mean photon numbers and interception rates.
For QPSK the idealized attack induces a QBER above 11% at about 0.65 mean photon number. 



### Updates

Since this work was done the attack speed has been improved, where the constraint solver was suboptimal before,
being limited to about 16k sifts for full key recovery, the demo can now run up to a million on a consumer laptop.

Subsequently, another attack has also been released against cascade (https://arxiv.org/abs/2603.29669).
 "The Manipulate-and-Observe Attack on Quantum Key Distribution" targets cleverly disturbing reconciliation for recovery.


### Assumptions

We will use a simplified model with the following parameters:
- Eve must obey the laws of quantum mechanics
- Eve must compute in polynomial time and memory (and so must Alice and Bob for error correction)
- We assume no side channels or imperfections in Alice and Bob’s equipment
- Alice never ~ amplitude 1.0 photons as weak coherent pulses. this attack differs than PNS though as it uses generalized measurement
- For simplicity, we assume a zero noise channel between Alice and Eve, and a zero noise channel between Eve and Bob.
- Under normal conditions when Eve is not present, Alice and Bob expect and accept an error rate threshold.
- Alice and Bob perform one-way 4-state BB84 with prepare & measure
- Alice and Bob do not use hardening measures for BB84 such as decoy states, randomization, and weaker (<1) coherent states of light
- Alice transmits single photon sources, with amplitude=1.0, using 4-QPSK states
- Alice and Bob use secure random number sources
- Alice and Bob’s classical channel is authenticated. While it is fully visible to Eve, it is not malleable to Eve and she can not recover the authentication keys to manipulate the classical channel.

### Background Literature
- [Original BB84 Publication](https://arxiv.org/abs/2003.06557)
- [Simple Proof of Security of the BB84 Quantum Key Distribution Protocol](https://arxiv.org/abs/quant-ph/0003004)
- [The Security of Practical Quantum Key Distribution](https://arxiv.org/abs/0802.4155)
- [Implementation Attacks against QKD Systems](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/Publications/Studies/QKD-Systems/QKD-Systems.pdf)
- [Quantum State Discrimination](https://arxiv.org/abs/0810.1970)
- [Phase randomization improves the security of quantum key distribution](https://arxiv.org/pdf/quant-ph/0504209)

## Prepare & Measure BB84 up to check bits

<img width="567" height="539" alt="image" src="https://github.com/user-attachments/assets/0885c8b0-dba7-4ba2-ad16-86a8e7bc903e" />

## QBER vs I&R 

Eve performs a quadrature measurement at the helstrom limit and re-transmits the qubits to Bob, with some error. Consider that at photon amplitude 1.0 the helstrom bound minimum error is approximately 0.092.  The branch diagram below shows the statistics for what Bob measures when his basis matches Alice’s


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
<img width="1499" height="1002" alt="image" src="https://github.com/user-attachments/assets/b3039b28-568a-4639-836d-8105f7323098" />

## Using the code

```
# grab cascade
git submodule update --init --recursive
# run the sims
PYTHONPATH=$PWD/cascade-python python3 plot_*py
```

## Attack Results Against adapative LDPC Scheme

See ldpc/, Eve is able to get 100% recovery at ~6% QBER
