---
title: "ml-intro"
teaching: 10
exercises: 0
questions:
- "None yet"
objectives:
- "Explain what we will cover in this workshop"
keypoints:
- "Something interesting about data science"
---

This is the introductory page.

## The crux of learning

Here I can put any infomation about what the course will cover.

![Forking Repositories]({{ page.root }}/fig/forking.svg)

Maybe include some images as well...

![FCN diagram]({{ page.root }}/fig/FCNDiagram.png)

![Repository Links]({{ page.root }}/fig/repository-links.svg)

## Helpful tools

If I want to have code-like text in the main text body I can do this `python take-over-the-world.py`

> ## Teaching Tools
>
> I can outline here some tools that may be useful, such as:
> [Jupyter Notebook][jupyter],
> Wow somehow carpentries knows how to link to jupyter without actually specifying the link. 
> I need to figure out how to include an actual link
{: .callout}

## LaTeX typesetting

LaTeX can be easily used with the usual double dollar signs $$y = mx + c$$, although this will appear on the same line.

 Some common definitions are included below.
 
  * **Data** $(x_i, y_i)$ where $i$ represents the $$i^{\text{th}}$$ data point. The $$x_i$$ are typically referred to as **instances** and the $$y_i$$ as **labels**. In general the $$x_i$$ and $$y_i$$ don't need to be numbers. For example, in a dataset consisting of pictures of animals, the $$x_i$$ might be images (consisting of height, width, and color channel) and the $$y_i$$ might be a string which states the type of animal. 

## Code presentation

I will present code as follows:

~~~
bash ./find-bsm-physics.sh
~~~
{: .language-bash}

~~~
python get_nobel_prize.py
~~~
{: .language-python}

~~~
Traceback (most recent call last):
  File "./get_nobel_prize.py", line 5, in <module>
    assert nobel_prize == True
AssertionError
~~~
{: .output}


> ## Some extra tips...
>
> It is better to code on a computer than on clay tablets
{: .callout}

## End of introduction

Wrap it up here.

{% include links.md %}
