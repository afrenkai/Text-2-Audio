from graphviz import Digraph


def create_transformer_tts_diagram():
    dot = Digraph(comment='Transformer TTS Architecture', format='pdf')

    dot.attr(rankdir='TB', size='8,5')
    dot.attr('node', shape='box', style='filled', fillcolor='lightgrey')

    dot.node('Text_Input', 'Text Input', shape='oval', fillcolor='lightblue')
    dot.node('Mel_Input', 'Mel Input\n(SOS + Mel Frames)', shape='oval', fillcolor='lightblue')

    dot.node('Encoder_PreNet', 'Encoder Pre-Net\n(Embedding + Linear + Conv1D Layers)')
    dot.edge('Text_Input', 'Encoder_PreNet')
    dot.node('Positional_Encoding_Enc', 'Positional Encoding')
    dot.edge('Encoder_PreNet', 'Positional_Encoding_Enc')
    encoder_blocks = ['Encoder_Block_1', 'Encoder_Block_2', 'Encoder_Block_3']
    for idx, block in enumerate(encoder_blocks):
        dot.node(block, f'Encoder Block {idx + 1}\n(Multi-Head Attention + Feed-Forward)')
        if idx == 0:
            dot.edge('Positional_Encoding_Enc', block)
        else:
            dot.edge(encoder_blocks[idx - 1], block)
    dot.node('Encoder_Output', 'Encoder Output')
    dot.edge(encoder_blocks[-1], 'Encoder_Output')
    dot.node('Decoder_PreNet', 'Decoder Pre-Net\n(Linear Layers)')
    dot.edge('Mel_Input', 'Decoder_PreNet')
    dot.node('Positional_Encoding_Dec', 'Positional Encoding')
    dot.edge('Decoder_PreNet', 'Positional_Encoding_Dec')
    decoder_blocks = ['Decoder_Block_1', 'Decoder_Block_2', 'Decoder_Block_3']
    for idx, block in enumerate(decoder_blocks):
        dot.node(block, f'Decoder Block {idx + 1}\n(Self-Attention + Encoder-Decoder Attention + Feed-Forward)')
        if idx == 0:
            dot.edge('Positional_Encoding_Dec', block)
        else:
            dot.edge(decoder_blocks[idx - 1], block)
        dot.edge('Encoder_Output', block, label='Encoder Output', style='dashed')
    dot.node('Mel_Projection', 'Linear Projection\n(Mel Spectrogram)')
    dot.edge(decoder_blocks[-1], 'Mel_Projection')

    dot.node('Stop_Token_Projection', 'Linear Projection\n(Stop Token)')
    dot.edge(decoder_blocks[-1], 'Stop_Token_Projection')
    dot.node('Post_Net', 'Post-Net\n(Conv1D Layers)')
    dot.edge('Mel_Projection', 'Post_Net')
    dot.node('Mel_Output', 'Output Mel Spectrogram', shape='oval', fillcolor='lightgreen')
    dot.edge('Post_Net', 'Mel_Output')

    dot.node('Stop_Token_Output', 'Stop Token Output', shape='oval', fillcolor='lightgreen')
    dot.edge('Stop_Token_Projection', 'Stop_Token_Output')
    dot.render('transformer_tts_architecture', view=True)


def create_seq2seq_tts_diagram():
    dot = Digraph(comment='Seq2Seq TTS Architecture', format='pdf')
    dot.attr(rankdir='TB', size='8,5')
    dot.attr('node', shape='box', style='filled', fillcolor='lightgrey')
    dot.node('Text_Input', 'Text Input', shape='oval', fillcolor='lightblue')
    dot.node('Mel_Spec_Input', 'Mel Spectrogram Input\n(SOS + Mel Frames)', shape='oval', fillcolor='lightblue')
    dot.node('Encoder_Embedding', 'Embedding Layer')
    dot.edge('Text_Input', 'Encoder_Embedding')
    dot.node('Encoder_Conv1', 'Conv1D Layer\n+ BatchNorm + ReLU')
    dot.edge('Encoder_Embedding', 'Encoder_Conv1')
    dot.node('Encoder_Conv2', 'Conv1D Layer\n+ BatchNorm + ReLU')
    dot.edge('Encoder_Conv1', 'Encoder_Conv2')
    dot.node('Encoder_BiGRU', 'Bi-GRU Layer')
    dot.edge('Encoder_Conv2', 'Encoder_BiGRU')
    dot.node('Encoder_Output', 'Encoder Output')
    dot.edge('Encoder_BiGRU', 'Encoder_Output')
    dot.node('Decoder_PreNet', 'Decoder Pre-Net\n(Linear Layers)')
    dot.edge('Mel_Spec_Input', 'Decoder_PreNet')
    dot.node('Location_Sensitive_Attention', 'Location-Sensitive Attention\nwith Location Layer')
    dot.node('Attention_RNN', 'Attention RNN\n(GRU Layer)')
    dot.edge('Decoder_PreNet', 'Attention_RNN')
    dot.edge('Encoder_Output', 'Location_Sensitive_Attention', style='dashed')
    dot.edge('Attention_RNN', 'Location_Sensitive_Attention')
    dot.node('Attention_Context', 'Context Vector')
    dot.edge('Location_Sensitive_Attention', 'Attention_Context')
    dot.node('Decoder_RNN', 'Decoder RNN\n(GRU Layer)')
    dot.edge('Attention_RNN', 'Decoder_RNN')
    dot.edge('Attention_Context', 'Decoder_RNN')
    dot.node('Mel_Projection', 'Linear Projection\n(Mel Spectrogram)')
    dot.edge('Decoder_RNN', 'Mel_Projection')
    dot.node('Stop_Token_Projection', 'Linear Projection\n(Stop Token)')
    dot.edge('Decoder_RNN', 'Stop_Token_Projection')
    dot.node('Post_Net', 'Post-Net\n(Conv1D Layers)')
    dot.edge('Mel_Projection', 'Post_Net')
    dot.node('Mel_Output', 'Output Mel Spectrogram', shape='oval', fillcolor='lightgreen')
    dot.edge('Post_Net', 'Mel_Output')
    dot.node('Stop_Token_Output', 'Stop Token Output', shape='oval', fillcolor='lightgreen')
    dot.edge('Stop_Token_Projection', 'Stop_Token_Output')
    dot.node('Attention_Weights', 'Attention Weights', shape='oval', fillcolor='lightgreen')
    dot.edge('Location_Sensitive_Attention', 'Attention_Weights', style='dashed')
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Text_Input')
        s.node('Mel_Spec_Input')

    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Mel_Output')
        s.node('Stop_Token_Output')
        s.node('Attention_Weights')

    dot.render('seq2seq_tts_architecture', view=True)

if __name__ == '__main__':
    create_seq2seq_tts_diagram()
    create_transformer_tts_diagram()
