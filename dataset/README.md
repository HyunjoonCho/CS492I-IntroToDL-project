## Dataset
Used two datasets  
Loaded stock price data every training using FinancialDataReader library  

### Category Dataset
Downloaded from [Final Project: 한국어 뉴스 기사 분류](http://ling.snu.ac.kr/class/cl_under1801/FinalProject.htm)  
* 8 categories: 정치(0), 경제(1), 사회(2), 생활/문화(3), 세계(4), 기술/IT(5), 연예(6), 스포츠(7)  
* 200 articles per category  

### Fluctuation Dataset
Crawled from Google News RSS  
* Filtered by Pre-trained News Category Classifier: only take 경제, 기술/IT  
* 4 categories: Big drop(<-2.5%, 0), Slight drop(<0%, 1), Slight jump(<2.5%, 2), Big jump(2.5%<, 3)
* 20,581 articles in total
* Skewed dataset, about 10% : 40% : 40% : 10%