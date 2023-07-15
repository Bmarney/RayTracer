Assignment 3 RayTracer

NAME: Brendan Marney

LANGUAGE:
-Python 
(-English)

LIBRARIES USED:
-Numpy
-Math
-Sys

I ran my code with "python RayTracer.py testAmbient.txt" in the command line. I included the key files and testXXXX.txt files as well just in case

I made my raytracer in Python, I believe that I implemented many of the parts correctly
with the exeption of the normals. I spent far too long trying to solve problems on this assignment
and I started it before the assignment was even released so the time was not an issue. I tried everything I 
could think of for the normals and sought help from the slides, my classmates and TAs but even though I feel 
that I understood what it was to take the inverse transform for the normals, my implementation never quite worked with it.
I put more effort into this assingment than I have almost any other in my time at uvic and I really think I got close with 
the normals but it just wasnt worth the stress and torture I was putting myself through so I am submitting what I have. 
My raytracer works well on about half of the test cases and the others are really close and just missing those normals.
I did have trouble with the diffuse and specular models especially implementing the R dot V for specular. every time I tried it just 
looked terrible and once again because of those damn normals!!! so I ended up using a blinn type method for it and that looked 
much better so I hope you like it!

I would also like to say that the marking scheme for this assignment feels really rough, the way it reads I wouldnt be surprised if I lost 
50% of my mark for just not quite getting the normals and with the shear amount of work I put in that feels pretty shitty. If you are feeling 
generous it would be nice to give out partial marks to students, like myself, who figured most of it out but not quite everything and even though it doesnt
produce the correct output. Even if that doesnt happen for this assignment I STRONGLY reccomend doing that for the next time the course is offered because that 
type of all or nothing marking makes every mistep feel hopeless because you need to figure out so much more to even get one mark. On the other hand partial marks 
encourage people to keep going after making great progress and slowly inching closer to the goal. It makes every small success feel like a huge milestone and builds 
confidence to keep going forward towards harder problems. I know this assignment was much more stressful for me and many students 
because of the all or nothing marking scheme

I talked to Grace Shorno and Matthew Dawson about high level concepts for the project; they really helped me with some of those concepts and I would have had 
a much worse time without them so give them a bonus mark if you can! I found one line of code on the internet and can't remember where from even:
"print("{:3.0f}%".format((float(j)/float(height) * 100)), end="\r")"
and showed it to Matthew who made it actually work and I think used it as well but it made waiting for the raytracer much more fun!

Hopefully this is everything, thanks for marking! Have a great break and a great next year!