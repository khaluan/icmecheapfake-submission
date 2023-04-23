# Detecting Cheapfakes using Self-query Adaptive-context learning

This is the source code for the paper [Detecting Cheapfakes using Self-query Adaptive-context learning](http://google.com)

## Getting started 
---
**1. Source code**: `$ git clone` this repo and install the Python dependencies from `requirements.txt`. Please refer to the docker [script](Dockerfile) for preparing the dependencies. 

You will also need to download the pre-trained model for [DocNLI](https://github.com/salesforce/DocNLI) paper from [here](https://drive.google.com/file/d/12kNONo0jgktxU0vWtV3Z2ZrCrB3DJPVj/view?usp=sharing) and put it in the `cache` folder.

**2. Dataset**: Please refer to the original [COSMOS dataset](https://github.com/shivangi-aneja/COSMOS). The dataset is assumed to be structured in the same way as in the `Dataset` folder, just simply replace the corresponding file/folder with the full version.

**3. Credential**: The code uses Google Cloud Vision API, therefore, you need to setup an API key to run the code. 

First, enable Google Cloud Vision API [here](https://console.cloud.google.com/apis/library/vision.googleapis.com)

Then set up the JSON credential for a service account. Refer to the following [tutorial](https://developers.google.com/workspace/guides/create-credentials#service-account) and stored it in the `cred.json` file in the main directory.

Notes: 
    
- Add your cred.json file into `.gitignore` list to prevent accidental commit of your key to the code repository. 
- The code uses the Google API only for reverse image search purpose, and it only used once during multiple experiments. Users are recommended to run the code once to collect the reverse search result (which is then stored in `Output` directory). For later runs, if the images are not changed, then commment these lines in the `Crawler\main.py` file to save the number of calls to the API.

```python
    with open(f'./Output/url_{task_name}.txt', 'w+', encoding='utf8') as file:
        for datapoint in tqdm(data):
            try:
                result = ''
                result = reverse_image_search_try_catch(datapoint['img_local_path'])
                result = preprocess_post(result, remove_irrelevant=True)
            except Exception as e:
                # print(e)
                pass
            file.write(str(result))
            file.write('\n')        

```

**4. Evaluation**: Run the `main.py` file in the main directory for evaluation result. The steps are well documented in the file.

**5. Build Docker image**: To build docker image, you need to first remove the `Dataset` folder as it will be used to evaluate the hidden test. Then refer to the Docker submission instructions [here](https://github.com/detecting-cheapfakes/detecting-cheapfakes-code).

**6. Run Docker**: Refer to the this [link](https://hub.docker.com/repository/docker/khaluan/icmecheapfakes/general) for instruction to run inference with custom dataset.