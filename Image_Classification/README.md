# AI Image Processing and Classification Project

---

## Final Report

---

### Summarize your findings from each part of the project.

**Setup -** The setup was the hardest part as I usually work in visual studio community and not in VS Code so the IDE was different from what I was used to but thankfully I was able to pick up quickly and set everything up in a decent amount of time.

**Base Classifier Section -** Code reviewing the “basic_classifier.py” file was simple enough. I prompted the AI to assist me in understanding each line and used the basic knowledge I have on python to understand it. It basically imports the tensor workflow library and messes around with its functionality to get a read on an image. It also sets the level of the model that predicts the images.

When generating the GradCam this gave me some issues as I found out the hard way that instead of opening a file through the IDE itself I had to do it through the terminal which was slightly frustrating to figure out. I had changed the file in the IDE but since I did not open it through the terminal the file wasn’t the correct one and the changes weren’t being applied, once I made the switch; however, the changes were applied and I was able to see the changes of the gradcam. Furthermore I had to iterate with the chosen AI that generated the GradCam so instead of opening a GUI window it would save directly on the repo folder.

**Basic Filter Section -** Reviewing the code was simple. I was able to figure it out without having to prompt the AI which I did anyway to see if I had missed something and try to understand it better. It basically imported the library image, and imageFilter to be able to edit images, it then created an image blur function which applied a gaussian blur to the image provided and saved the image as the output.
When generating my own image filter I decided to go with a vignette around the borders and image noise to increase the focus of the image on the chosen image. I prompted AI to create this and all I had to do here to collaborate with the AI was tweak the values in the function to the values I wanted to create the filter I had imagined.

---

### Highlight what you learned about the classifier's behavior from the heatmap.
The biggest thing I learned is that the heatmap focuses on the center of the image provided which in my case was of a dog. And if it pin points a face it will give most of its focus on that as well as anything that is connected to it.

---

### Describe the filter you developed. What kind of effect does it have on your image?
The filter I developed was a vignette around the image borders and noise. It makes the image softer and places more focus on the dog. The effect provided is more of a 1990s camera with very bold vignette.

---

### Reflect on your experience working with the AI to explain and write Python code.
The experience in working with AI to explain and write Python code was helpful, prompting it to explain the code line by line gave me a better understanding of the file I was looking at as a whole, and it broke it down into very simple terms. With this I was able to prompt it to write code to apply effects or features I wanted the program to do and I was able to implement, tweak, and change the file as I saw fit to get the results that were asked in the assignment.

