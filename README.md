# LA-LLAMA3

LA-LLAMA3 is a minimalist implementation of the LLAMA3 algorithm, primarily using linear algebra operations,
and its main program is confined to just 100 lines of code. This project serves as a tool to provide a clear
understanding of the fundamental workings of the LLAMA3(and generally, Transformer) architecture.

And it actually works.

This branch includes some additional code for the KV cache, and experiments with using float8 to store the KV cache.

## 2 bit K Cache and 4 bit V Cache

This branch experiments with DCT to reduce K cache to 2 bits per value, and V cache to 4 bits per value, which is a 10x reduction in memory usage.

## And it actually works

For example,

```
So God created mankind in his own image, in the image of God he created them; male and female he created them.
```

Is generated from

```
So God created mankind in his own image, in the image of God
```

With 2 bit K cache and 4 bit V cache!

## Comparison

This compression is done per token separately with fixed length per token.

So it can be efficient implemented with PagedAttention, without the need of full reload/save like
other cache compression methods, e.g. [KIVI](https://arxiv.org/abs/2402.02750)(which uses per channel compression).
