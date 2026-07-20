# E8 historical closure blob inspection

- raw bytes: `10018`
- raw SHA-256: `f63ad017feee1a06926a92f27446c123b80df9bb79ffb12a71e3865cc948b94a`
- compact base64 characters: `10017`
- compact length mod 4: `1`
- invalid character count: `8`
- first invalid characters: `[(10000, '[', 91), (10001, '.', 46), (10002, '.', 46), (10003, '.', 46), (10013, '.', 46), (10014, '.', 46), (10015, '.', 46), (10016, ']', 93)]`
- first 120 characters: `H4sICFCVV2oCA2U4X2ZpbmFsLnBhdGNoAOxae3MTV5b/n09xl6rd2Y1pqVstdUuuZStCbmwVsuToYcLMpORW921bE0mt7W4BnlSq7BCDAT9IwvuRhBAeQ7DN`
- last 240 characters: `NIOoZDYXwfM8sG3ti6yC24Ku3tkX1bkdGpJPfK1oOWnoO/mAtosfWgaeTCI2VWKgS5JGFAp+TVyjdyAgqNXCofX45SXZWkvkojDujbbEPb8eF0a/tqCnU8sXYcUKwvV2eFImUJY4XqmlvjGCRYuMCUvb0qTlpeudQFGdAppJarikoW2ZGcOmr5OXJYu8LTqxG83EQ3a27B8nK4r7mYSlk6W7mAjbKg[...truncated...]
`
- strict decode: `FAIL Error: Only base64 data is allowed`
