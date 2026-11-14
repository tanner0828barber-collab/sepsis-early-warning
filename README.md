# Sepsis Early Warning: does it travel?

**This project is a prototype and is not medical advice.**

An entry for the Veritas AI International High School Hackathon, Health track, built solo in the 48 hour window that opened on Sat 11/14/2026.

## The question

Can a sepsis early warning model trained at one hospital travel to another? We train on public intensive care data from hospital A, freeze the pipeline and the alert threshold, and evaluate once on hospital B.

## Status

Planning files only. No model has been trained yet. Results will be added here, including weak ones.

## Read these first

- [SPEC.md](SPEC.md): the one page plan
- [DEFINITIONS.md](DEFINITIONS.md): what "caught," "missed," "false alarm," and "onset" mean here, and the disclosure about hospital B
- [DECISIONS.md](DECISIONS.md): every choice, with the time and the reason, and the prediction written before any model was trained

## Data

PhysioNet / Computing in Cardiology Challenge 2019, training sets A and B. Public, de-identified, license CC BY 4.0. https://physionet.org/content/challenge-2019/1.0.0/

The data is not stored in this repository. The notebook downloads it from the public source.

## Credits

Dataset and challenge: Reyna et al., "Early Prediction of Sepsis from Clinical Data: the PhysioNet/Computing in Cardiology Challenge 2019." Full credits, libraries, and AI tools used will be listed in the write-up.
