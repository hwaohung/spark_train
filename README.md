# spark_train

1 因為找不太到spark提供的multiclass SVM，所以自己做簡單的版本，training時主要就是對每個class產生一個分類器，
  test時，就把資料丟給每一個分類器，若label是1則預測的結果為該分類器所隸屬的label．若有多個分類器結果都是1，
  就採用training data label的分佈數量最多的來決定這個不確定的label，若都沒有分類器的結果是1，則將此label設為
  training data label的分佈數量最少的label．

2 此題想到的方法是用先用PCA，對資料做降維，之後再用SVM對這個被降為的資料做training & test


因為執行時遇到一些問題（記憶體size），還在處理中，會盡快放上來．
