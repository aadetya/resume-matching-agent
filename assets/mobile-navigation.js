const buttons = [...element.querySelectorAll('button[data-pane]')];
function activate(button, focus = false) {
  const layout = document.querySelector('#review-layout');
  if (!layout) return;
  layout.dataset.pane = button.dataset.pane;
  buttons.forEach(item => {
    item.setAttribute('aria-selected', String(item === button));
    item.tabIndex = item === button ? 0 : -1;
  });
  if (focus) button.focus();
}
element.addEventListener('click', event => {
  const button = event.target.closest('button[data-pane]');
  if (button) activate(button);
});
element.addEventListener('keydown', event => {
  const index = buttons.indexOf(event.target);
  if (index < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const target = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 :
    (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
  activate(buttons[target], true);
});
