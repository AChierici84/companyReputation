# Monitoraggio della reputazione online di un’azienda


Per effettuare monitoraggio sulla reputazione di online di un'azienda verrà valutato il seguente modello:
[https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest](https://https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

Twitter-roBERTa-base è un modello svilupato per effettuare Sentiment Analysis.
Il training iniziale è stata fatto con tweet dal 2018 al 2022, in inglese. 
Viene utilizzato come base di partenza per fare fine tuning su task similari come sentment analysis /classificazione. 
È un modello della famiglia RoBERTa (Robustly Optimized BERT Approach) una versione migliorata di BERT sviluppata da Facebook.

Le label utilizzate nel pre-training sono:
* 0 > Negative
* 1 > Neutral
* 2 > Positive

Come dataset per il fine tuning verrà utilizzato SemEval Twitter Sentiment, una raccolta di Tweet annotati con etichette di sentiment, creato per le competizioni SemEval (Semantic Evaluation). 
Le etichette sono:

*   0 = negative
*   1 = neutral
*   2 = positive

equiparabili a quelle originali del modello. Questo semplifica l'elaborazione perchè non è necessario un mapping.
La distribuzione dele dataset è la seguente:
||Train	|Test	|Validation	|Total|
|---:|---:|---:|---:|---:|
|Negative|7093|3972|312	|11377|
|Neutral|20673|5937|869	|27479|
|Positive|17849|2375|819	|21043|
|Total|45615|12284|2000|59899|

Le classi sono leggermente sbilanciate. La classe più numerosa è quella dei tweet neutri, a seguire i tweet positivi , infine i tweet negativi. 
La distribuzione comunque è stata rispetta nella divisione tra train, test e validation.

Una volta caricato il modello pre-trained, tokenizziamo il dataset imponendo un limite alla lunghezza dei tweet (Es. 180)
Sono state attuate le seguenti strategie di training:
* validaziane alla fine di ogni epoca
* salvataggio checkpoint
* learning rate inziiale pari a 2e-5
* batch size di training/validazione di 16
* numero di epoche 3
* regolarizzazione con riduzione dei pesi
* valutazione del modello migliore a fine training
* metriche aggiuntive accuracy e F1

Es. Log di training

|Epoch|Training Loss|Validation Loss|Accuracy|F1|
|---:|---:|---:|---:|---:|
|1|0.555000|0.548251|0.753500|0.741171|
|2|0.425300|0.601499|0.768500|0.751958|
|3|0.263600|0.722670|0.764500|0.753936|

La valutazione del modello sul test set dava inizialmente:

{'eval_loss': 0.6049224138259888, 'eval_accuracy': 0.7307880169325952, 'eval_f1': 0.7303927353156595, 'eval_runtime': 80.6559, 'eval_samples_per_second': 152.301, 'eval_steps_per_second': 9.522, 'epoch': 3.0}

La matrice di confusione rivela una parziale difficoltà di classificazione tra classi "adiacenti" Es. tra negativo e neutro o tra positivo e neutro.
Sono stati valutati inoltre alcuni test tra quelli mal classificati per tentare di individuare la ragione degli errori. 

In alcuni casi espressioni come "it isn't too bad" traggono il sistema in inganno, in altri invece trovo la mood negativa sia accettabile in presenza di termini come "lost", "anything". A volte il tono del tweet nel complesso può essere neutro pur avendo alcuni termini negativi, non sempre è semplice per il modello raggiungere questo grado di ragionamento.


Il modello realizzato seppure non raggiunge prestazioni ottime si attesa su un discreto 73% e costituisce una buona base per l'inferenza. 
Sarà utile aggiungere tweet più mirati sull'azienda scelta per aumentare le prestazioni del modello.

Il modello realizzato è stato salvato su hugging fase ed è disponibile al seguente link
[https://huggingface.co/AChierici84/sentiment-roberta-finetuned](https://huggingface.co/AChierici84/sentiment-roberta-finetuned)

Sono stati realizzati quindi i seguenti moduli python:
* Crawling module : scaricamento di tweet di un account X di customer support Es. @AmazonHelp
* Analysis module : analisi dei tweet scaricati tramite il modello
* Feedback module : feedback utente per confidence inferiore al 80%
* Training module : modulo di retrain con feedback
* Testing module : test di integrazione, test di raggiungilità dei componenti necessari al crawling e all'analisi, test sulla distribuzione dati

Sono state inoltre predisposte le seguenti azioni GitHub:
* Crawl : eseguita ogni mattina alle ore 8.00 scarica i nuovi tweet inviati all'account di customer support
* Analysis : anlizza i nuovi tweet scaricati (esegue un'ora d'opo il task precedente)
* Continous integration : esecuzione unit test e integration test ad od ogni push/pull
* Training and continuous deployement: retraining con feedback e nuovi tweet eseguita ogni giorno alle 00:00

Api di inference disponibile su spaces:
[https://huggingface.co/spaces/AChierici84/companyReputation](https://huggingface.co/spaces/AChierici84/companyReputation)

Infine sono state implementate dashboard per il monitoraggio del training e la reputation.

* Dashboard reputation
In questa dashboard troviamo il conteggio dei tweet scaricati (nel range temporale selezionato), alcuni Sample di tweets scaricati, la reputation nell'intervallo selezionato,
la confidence media del modello per le analisi.

<img width="1076" height="692" alt="Screenshot 2025-11-26 091825" src="https://github.com/user-attachments/assets/51c73839-f48b-4c31-8235-57d7b07b1ccc" />

Un grafico in linea temporale , poi , riporta i tweet scaricati per giorno. E gli ultimi tweet negativi riscontrati nell'intervallo selezionato. 

<img width="1093" height="742" alt="Screenshot 2025-11-26 091839" src="https://github.com/user-attachments/assets/1cdc3215-ac81-496c-a3f7-e0054187d19c" />

* DashBoard training
In questa dashboard troviamo il dettgalio sui training eseguiti per il fine tuning sui feedback. Sono riportati l'ultima accuracy del modello e le statistiche dell'ultimo training.
Nella tabella è visibile il dettaglio dei dati sull'ultimo training.

<img width="1338" height="687" alt="Screenshot 2025-11-26 145739" src="https://github.com/user-attachments/assets/c0a9c6fc-0762-4f0d-adec-3437995ff7ee" />

Di seguito è riportata l'andamento della loss e dell'accuracy, il training time e i tempi medi di inferenza.

<img width="1333" height="532" alt="Screenshot 2025-11-26 145805" src="https://github.com/user-attachments/assets/c7bd26d2-1de9-4fa7-b390-63857644c859" />


  
 


