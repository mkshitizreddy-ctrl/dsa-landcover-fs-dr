# Notes

## T3: Guyon and Elisseeff (2003), An Introduction to Variable and Feature Selection, JMLR 3, 1157-1182

**Method:** Surveys feature selection (variable ranking, wrapper/embedded subset selection, feature construction such as PCA/SVD, validation) and gives a 10-step practical checklist.

**Finding:** Ranking features individually is simple, scalable and a good baseline, but it can keep redundant features and miss features that are only useful together, so subset selection is needed.

**Caution for our project:** High correlation does not always mean redundancy (Sec. 3.2), and a feature useless alone can help with others (Sec. 3.3). Our correlation-threshold removal can drop complementary features. Cover this in limitations and test it in the threshold-sensitivity study.
