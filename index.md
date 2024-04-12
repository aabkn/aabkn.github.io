# About me

I am a second-year PhD student at ETH Zurich working with Prof. Afonso Bandeira at Mathematics Department.
My research interests include
high-dimensional statistics, computational complexity, statistical physics, and stochastic optimization.
In particular, I work on understanding of phase transitions and
computational hardness of statistical inference.

Here is a link to my full [CV](/files/CV_Apr24_Kireeva.pdf).

# Publications
<ul>
  <li> Anastasia Kireeva, Joel Tropp (2024)
  <a href="https://arxiv.org/abs/2402.17873">Randomized matrix computations: Themes and variations</a>
  </li>
  <li> Ilyas Fatkhullin, Anas Barakat, Anastasia Kireeva, Niao He (2023)
  <a href="https://arxiv.org/abs/2302.01734.pdf">Stochastic Policy Gradient Methods:
  Improved Sample Complexity for Fisher-non-degenerate Policies </a>
  In: International Conference on Machine Learning. PMLR, pp. 9827–9869
  </li>
  <li> Anastasia Kireeva, Jean-Christophe Mourrat (2023)
  <a href="http://arxiv.org/abs/2304.05129">Breakdown of a concavity property of mutual information for non-Gaussian channels </a>
  In: Information and Inference: A Journal of the IMA 13.2
  </li>
</ul>

# Teaching

<ul>
  <li>
  Autumn semester of 2023:

  Course Coordinator - Mathematics of Data Science
  </li>
  <li>
  Spring semester of 2023 and 2024:

  Teaching assistant - Mathematics of Signals, Networks, and Learning Category
  </li>

  <li>
  Autumn semester of 2022:

   Teaching assistant - Mathematics of Data Science
  </li>
</ul>

# Posts
<ul>
  {% for post in site.posts %}
    <li>
      <span>{{post.date | date: "%b, %Y"}}</span>
      	<a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
