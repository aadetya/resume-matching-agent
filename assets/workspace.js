() => {
  // Run after Gradio paints the changed conversation. Move only the inner
  // log to the first line of the response; preserve the page position.
  setTimeout(() => {
    const chat = document.querySelector('#conversation');
    const log = chat?.querySelector('[role="log"]');
    const replies = chat?.querySelectorAll('[data-testid="bot"]');
    if (!log || !replies?.length) return;
    const reply = replies[replies.length - 1];
    log.scrollTop += reply.getBoundingClientRect().top - log.getBoundingClientRect().top - 12;
  }, 80);
}
