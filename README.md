## Optical Character Recognition
Extract text from images and classify accordingly. 

### Assumptions
* Letters are reasonably spaced. 
* Letters should reasonably resemble the training data set. 
* Image should be naturally decipherable i.e free from any occlusions and obfuscations.

### Output 
![Output](https://raw.githubusercontent.com/AND2797/Optical_Character_Recognition-/master/Output.png)
### Observations
* Common Confusions -
  * True Label 'C' -> Predicted Label 'G'
  * True Label '1' -> Predicted Label - '|' (I)
  * True Label '4' -> Predicted Label - 'A'
  * True Label 'O'/ '0' -> Predicted Label - 'O'/'0'/'Q'
  * True Label '8' -> Predicted Label 'B'
  * True Label '6' -> Predicted Label 'G'
  * True Label '9' -> Predicted Label 'g' (single storey) 
  
  * Most of the confusions share highly similar features and construction, for example 4 and A, 8 and B, even C and G to some extent. 
### Tasks
- [X] Train in PyTorch (EMNIST - Balanced)
- [ ] Refactor training loop [On - Hold] 
- [ ] Hyperparameter Study  [On - Hold]
- [X] Extract text from images
- [X] Pre-process to match EMNIST 
- [X] Test on extracted
- [X] Improve accuracy and pre-processing
- [ ] Try realtime OCR from video [On - Hold]
- [ ] Improve to make rotationa and scale invariant. [On - Hold]
