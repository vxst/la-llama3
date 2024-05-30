# LA-LLAMA3

LA-LLAMA3 is a minimalist implementation of the LLAMA3 algorithm, primarily using linear algebra operations,
and its main program is confined to just 100 lines of code. This project serves as a tool to provide a clear
understanding of the fundamental workings of the LLAMA3(and generally, Transformer) architecture.

And it actually works.

(Well, it is intend to generate one token, so I skipped things like real batching(it tied 2 sentences rather
with stacking, but it's single token generate and not intend to use KV cache anyway)