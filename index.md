# Home

Posts
<ul>
  {% for post in site.posts %}
    <li>
      <span>{{post.date | date: "%b %-d, %Y"}}</span>
      <h2>
      	<a href="{{ post.url }}">{{ post.title }}</a>
  	  </h2>
    </li>
  {% endfor %}
</ul>
