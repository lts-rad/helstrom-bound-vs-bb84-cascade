# LDPC Attack Demo

This demonstrates a 100% key recovery against an adaptive LDPC scheme. 

It's unclear if this LDPC application is correct as a substantial amount of bits are disclosed.

Code taken from https://github.com/marcoavesani/QKD_LDPC_python/tree/master based on https://github.com/RQC-QKD-Software

## Assumptions
- During the adaptive disclosure, Alice directly discloses the requested bit value to Bob
- Alice makes the extended vector information about puncturing, short codes public

## Attack Model
- This demonstrates eve performing quadrature measurement at the helstrom bound for a=1.0
- Next, eve uses the disclosed error bits and the syndrome information to run her own LDPC correction
- This allows eve to fully recover alice's key for the two code schemes provided


