# Use Case / Value add/ Benefit:
Generation 1:  MMP internal usage 
	- Pre-evaluate Customer requirements based on: 
		○ Previous customer requirements evaulations
		○ Product specification (e.g. TCD, SYS2, SYS3, SWE…)
		
	- Benefit: 

Generation 2: Bosch internal usage 

Generation 3: Global  usage 

# Application level: Target state 
	1) Input: 
		a. Upload new customer specificaition 
	2) Output: 
		a. Evaluation of these customer specification 
			i. Quantitative measure (fullfillment grade)
				1) Has it been evaluated before? Known requirement (yes/no) 
				2) Rating probability (accepted, rejected, partly accepted….)
			ii. Visual representation 



# Development level/approach: 

	- Development of training database: 
		○ Input data source 
			§ Requirement specification from customer incl. MMP evaluation 
			§ Technical Specification MMP  
			
		○ Input format 
			§ PDF,
			§ Excel 
			§ Word 
			§ Reqif 
		
		○ Target format: 
			§ Standardized database 
			
Choice of model :

| Modeltype | Key feature | Pro|
|----------|-------------|---------|
BERT | Pre-trained transformer | Easy to use
RoBERTa |	Like BERT |better training	Higher accuracy
DistilBERT|Smaller, faster BERT |	Speed, lower resource usage|
ALBERT|	Lite BERT, efficient|	Less memory, similar accuracy|
XLNet	|Permutation-based, long deps|	Handles longer context|
Electra|	Replaced token detection	|Efficient, strong performance|
DeBERTa	|Enhanced attention	|Top accuracy on benchmarks|

# Next steps
## Next steps Organization: 
	- Setup Github repository,   R: Steffen


## Next steps Pre-processing: 
	- Define input database based on available data , R: Mithun 
		○ First input: 
			§ Taking MMP2.11 - SW module 
			§ ReqIF export 
		
	- Define pre-processing (OCR) for conversion of input database in unified format 
	- Define target database format as input into the training phase 

## Next steps Model development: 
	- Define suitable pre-trained model for text classification, R: Steffen
		○ LSTM 
		○ Encoder/Decoder -> Autocoder 
		○ Transformer..
	Current status 01.08.25: Try with Bert Model 

# Open questions: 
	- Check approach from MIDAS and ETAS, R: Steffen
