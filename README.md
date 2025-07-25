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
			§ Technical Spefication MMP  
			
		○ Input format 
			§ PDF,
			§ Excel 
			§ Word 
			§ Reqif 
		
		○ Target format: 
			§ Standardized database 
			
	- Choice of model 
	- Training and validation of model 


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

# Open questions: 
	- Check approach from MIDAS and ETAS, R: Steffen
