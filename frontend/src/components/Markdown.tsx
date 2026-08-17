/** Renders assistant messages as markdown.
 *
 *  The model answers in markdown — headings per day, bold totals, bulleted stop lists — and shown
 *  as plain text that arrives as literal `###` and `**` noise.
 *
 *  react-markdown renders to React elements rather than an HTML string, so model output cannot
 *  inject markup — no raw-HTML injection anywhere in this path. Raw HTML in the source is left
 *  un-parsed by default, which is the behaviour we want.
 */

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface Props {
  children: string
}

export function Markdown({ children }: Props) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // The bubble is already a small surface, so headings step down rather than shout.
          h1: ({ children: c }) => <h4 className="md__h">{c}</h4>,
          h2: ({ children: c }) => <h4 className="md__h">{c}</h4>,
          h3: ({ children: c }) => <h4 className="md__h">{c}</h4>,
          h4: ({ children: c }) => <h4 className="md__h">{c}</h4>,
          a: ({ href, children: c }) => (
            // Model-supplied links are untrusted: never hand the opener a window reference.
            <a href={href} target="_blank" rel="noopener noreferrer nofollow">
              {c}
            </a>
          ),
          table: ({ children: c }) => (
            <div className="md__scroll">
              <table>{c}</table>
            </div>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
