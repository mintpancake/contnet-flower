# ConTNet Flower
An implementation of ConTNet on the Oxford 17 Category Flower Dataset

## Useful Statistics
training set mean: `[0.4256, 0.4259, 0.2790]`
training set std: `[0.2758, 0.2501, 0.2555]`

## Data Preparation
- Train(90%)/Test(10%) split:
  1. Extract dataset into `data/flowers/raw`
  2. Run `data/flowers/raw/_split.py`
  3. Run `data/flowers/raw/_generate_meta.py`
  - Please double check with me that our splits are identical

## Dev Instructions
- Workflow:
  1. Checkout a new branch from `main` (Do not work on `main`)
  2. Write code on your branch
  3. Pull `main` and rebase your branch on `main`
  4. Solve any conflicts
  5. Push your branch
  6. Create a pull request on GitHub
  7. Inform others of your work
- Dependencies: 
  - Follow HKU CS GPU Farm Tutorial page 8
  - See ./env.yaml
- Clone issue: 
  - See [创建个人访问令牌](https://docs.github.com/cn/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
