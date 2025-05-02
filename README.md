# OFI-feature-construction

Regarding the construction of the Cross-Asset OFI feature, I encountered some challenges. Specifically, in the paper “Cross-impact of Order Flow Imbalance in Equity Markets,” the concept of Cross-Asset OFI is introduced only within the context of a regression framework that includes the OFIs of multiple assets. The paper does not provide a standalone method for calculating a scalar Cross-Asset OFI feature at each timestamp. Moreover, the dataset provided for this task contains only a single asset, which makes it infeasible to perform a true cross-asset regression.

As a result, I adapted my approach by simulating a second asset based on the given data in order to approximate a cross-asset structure. Using this setup, I implemented the CII model as described in the paper and produced the corresponding regression summary.

Given that the simulated setup involved only two assets, I opted for OLS regression rather than LASSO, as the dimensionality of the explanatory variables did not require regularization or feature selection.

I apologize that I was unable to provide a definitive Cross-Asset OFI value per timestamp. Unfortunately, I could not locate any channel to contact the hiring team directly. If further clarification is needed, please feel free to reach out to me.
