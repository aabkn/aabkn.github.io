# About me

I am a first-year PhD student at ETH Zurich working with Afonso Bandeira at Mathematics Department.
My research interests include
high-dimensional statistics, computational complexity, statistical physics, and stochatic optimization.
In particular, I work on understanding of phase transitions and
computational hardness of statistical inference.

Here is link to my full [cv](/files/CV_Kireeva.pdf).

# Publications
<ul>
  <li> Ilyas Fatkhullin, Anas Barakat, Anastasia Kireeva, Niao He (2023)
  <a href="https://arxiv.org/abs/2302.01734.pdf">Stochastic Policy Gradient Methods:
  Improved Sample Complexity for Fisher-non-degenerate Policies</a>
  </li>
</ul>

# Teaching

<ul>
  <li>
  <span>Spring semester of 2023</span>Teaching assistant - Mathematics of Signals, Networks, and Learning Category
  </li>

  <li>
  <span>Autumn semester of 2022</span>Teaching assistant - Mathematics of Data Science
  </li>
</ul>

# Posts
<ul>
  {% for post in site.posts %}
    <li>
      <span>{{post.date | date: "%b %-d, %Y"}}</span>
      	<a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
